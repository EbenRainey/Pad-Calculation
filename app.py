import streamlit as st
import ezdxf
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from scipy.optimize import brentq
import io
import tempfile

# === Processing Functions ===

def extract_pad_centers(doc, layer_name="PAD CENTRE POINTS"):
    msp = doc.modelspace()
    centers = []
    for pt in msp.query(f'POINT[layer=="{layer_name}"]'):
        loc = pt.dxf.location
        centers.append((loc.x, loc.y, getattr(loc, 'z', 0.0)))
    return centers


def extract_tin_faces(doc, layer_name="TIN SURFACE"):
    msp = doc.modelspace()
    faces = []
    for face in msp.query(f'3DFACE[layer=="{layer_name}"]'):
        verts = [
            (face.dxf.vtx0.x, face.dxf.vtx0.y, face.dxf.vtx0.z),
            (face.dxf.vtx1.x, face.dxf.vtx1.y, face.dxf.vtx1.z),
            (face.dxf.vtx2.x, face.dxf.vtx2.y, face.dxf.vtx2.z)
        ]
        faces.append(np.array(verts))
    return faces


def generate_pad_polygons(centers, length, width):
    pads = []
    for x, y, _ in centers:
        dx = length / 2.0
        dy = width / 2.0
        corners = [(x - dx, y - dy), (x + dx, y - dy), (x + dx, y + dy), (x - dx, y + dy)]
        pads.append(Polygon(corners))
    return pads


def fit_plane(points):
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    b = points[:, 2]
    a, b_slope, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return a, b_slope, c


def enforce_slope(a, b_slope, min_s, max_s):
    grad = np.array([a, b_slope])
    mag = np.linalg.norm(grad)
    target = np.clip(mag, min_s, max_s)
    if mag > 0:
        grad *= (target / mag)
    return grad[0], grad[1]


def volume_difference(c_shift, a, b_slope, pts):
    diffs = (a * pts[:, 0] + b_slope * pts[:, 1] + c_shift) - pts[:, 2]
    cut = np.sum(diffs[diffs > 0])
    fill = -np.sum(diffs[diffs < 0])
    return cut - fill


def balance_cutfill(a, b_slope, points):
    func = lambda c: volume_difference(c, a, b_slope, points)
    try:
        c_balanced = brentq(func, -100.0, 100.0, xtol=1e-3)
    except ValueError:
        c_balanced = 0.0
    return c_balanced


def process_dxf(file_bytes, params):
    # Write bytes to temporary .dxf file
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    doc = ezdxf.readfile(tmp_path)

    centers = extract_pad_centers(doc)
    tin_faces = extract_tin_faces(doc)
    pads = generate_pad_polygons(centers, params['pad_length'], params['pad_width'])

    summary = []
    # Create output DXF
    out_doc = ezdxf.new('AC1032')
    out_msp = out_doc.modelspace()

    for idx, pad in enumerate(pads):
        pts = []
        for face in tin_faces:
            for x, y, z in face:
                if pad.contains(Point(x, y)):
                    pts.append((x, y, z))
        pts = np.array(pts)
        if pts.size == 0:
            continue

        # Fit plane and enforce slope
        a0, b0, _ = fit_plane(pts)
        a1, b1 = enforce_slope(a0, b0, params['min_slope'], params['max_slope'])
        c1 = balance_cutfill(a1, b1, pts)

        # Compute cut/fill
        diffs = (a1 * pts[:, 0] + b1 * pts[:, 1] + c1) - pts[:, 2]
        cut = float(np.sum(diffs[diffs > 0]))
        fill = float(-np.sum(diffs[diffs < 0]))

        summary.append({
            'pad_index': idx,
            'center_x': centers[idx][0],
            'center_y': centers[idx][1],
            'slope_a': a1,
            'slope_b': b1,
            'elevation_c': c1,
            'cut_volume': cut,
            'fill_volume': fill
        })

        # Add pad plane as a quad face
        coords = list(pad.exterior.coords)[:-1]
        z_vals = [a1 * x + b1 * y + c1 for x, y in coords]
        # Define four vertices
        v1 = (coords[0][0], coords[0][1], z_vals[0])
        v2 = (coords[1][0], coords[1][1], z_vals[1])
        v3 = (coords[2][0], coords[2][1], z_vals[2])
        v4 = (coords[3][0], coords[3][1], z_vals[3])
        out_msp.add_3dface([v1, v2, v3, v4])

    # Write DXF to string buffer, then encode to bytes
    str_buffer = io.StringIO()
    out_doc.write(str_buffer)
    dxf_str = str_buffer.getvalue()
    dxf_bytes = dxf_str.encode('utf-8')

    df = pd.DataFrame(summary)
    return dxf_bytes, df

# === Streamlit UI ===
st.title("Pad Design & Cut/Fill Balance")
st.markdown("Upload a DXF containing layers 'PAD CENTRE POINTS' and 'TIN SURFACE', then set parameters below.")

uploaded = st.file_uploader("Upload DXF file", type=["dxf"])

st.sidebar.header("Design Parameters")
pad_length = st.sidebar.number_input("Pad length (m)", min_value=1.0, max_value=100.0, value=10.0)
pad_width  = st.sidebar.number_input("Pad width (m)", min_value=1.0, max_value=100.0, value=10.0)
min_slope  = st.sidebar.slider("Min pad slope (%)", min_value=0.0, max_value=10.0, value=0.5)/100
max_slope  = st.sidebar.slider("Max pad slope (%)", min_value=0.0, max_value=10.0, value=2.0)/100
batter_min = st.sidebar.slider("Min batter slope (1:X)", min_value=1, max_value=10, value=2)
batter_max = st.sidebar.slider("Max batter slope (1:X)", min_value=1, max_value=10, value=5)
max_b_len  = st.sidebar.number_input("Max batter length (m)", min_value=0.1, max_value=50.0, value=4.0)
tolerance  = st.sidebar.number_input("Cut/Fill tolerance (m³)", min_value=0.0, max_value=100.0, value=10.0)

if uploaded:
    params = {
        'pad_length': pad_length,
        'pad_width': pad_width,
        'min_slope': min_slope,
        'max_slope': max_slope,
        'batter_min_slope': 1 / batter_min,
        'batter_max_slope': 1 / batter_max,
        'max_batter_length': max_b_len,
        'tolerance': tolerance
    }
    dxf_bytes, summary_df = process_dxf(uploaded.read(), params)
    st.download_button("Download Design DXF", data=dxf_bytes, file_name="design_output.dxf")
    st.write("### Volume Summary")
    st.dataframe(summary_df)
    csv_data = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Summary CSV", data=csv_data, file_name="summary.csv")
