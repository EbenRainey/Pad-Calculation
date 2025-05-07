import streamlit as st
import ezdxf
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from scipy.optimize import brentq
import io
import tempfile

# === Geometry & Volume Helpers ===
def triangle_area(pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))) / 2


def extract_pad_centers(doc, layer_name="PAD CENTRE POINTS"):
    return [(pt.dxf.location.x, pt.dxf.location.y, getattr(pt.dxf.location, 'z', 0.0))
            for pt in doc.modelspace().query(f'POINT[layer=="{layer_name}"]')]


def extract_tin_faces(doc, layer_name="TIN SURFACE"):
    faces = []
    for face in doc.modelspace().query(f'3DFACE[layer=="{layer_name}"]'):
        verts = [
            (face.dxf.vtx0.x, face.dxf.vtx0.y, face.dxf.vtx0.z),
            (face.dxf.vtx1.x, face.dxf.vtx1.y, face.dxf.vtx1.z),
            (face.dxf.vtx2.x, face.dxf.vtx2.y, face.dxf.vtx2.z)
        ]
        faces.append(np.array(verts))
    return faces


def generate_pad_polygons(centers, length, width):
    pads = []
    half_x, half_y = length/2.0, width/2.0
    for x, y, _ in centers:
        pads.append(Polygon([
            (x-half_x, y-half_y), (x+half_x, y-half_y),
            (x+half_x, y+half_y), (x-half_x, y+half_y)
        ]))
    return pads


def fit_plane(points):
    A = np.column_stack((points[:,0], points[:,1], np.ones(len(points))))
    b = points[:,2]
    a, b_slope, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return a, b_slope, c


def enforce_slope(a, b_slope, min_s, max_s):
    grad = np.array([a, b_slope])
    mag = np.linalg.norm(grad)
    if mag > 0:
        target = np.clip(mag, min_s, max_s)
        grad *= (target / mag)
    return grad[0], grad[1]


def volume_stats_plane(a, b_slope, c, faces, polygon):
    cut = fill = 0.0
    for tri in faces:
        centroid = tri[:,:2].mean(axis=0)
        if not polygon.contains(Point(*centroid)):
            continue
        area = triangle_area(tri[:,:2])
        zg = tri[:,2].mean()
        zp = a*centroid[0] + b_slope*centroid[1] + c
        dv = zp - zg
        if dv > 0: cut += dv * area
        else: fill += -dv * area
    return cut, fill


def balance_cutfill(a, b_slope, faces, polygon, tol):
    fn = lambda c: (volume_stats_plane(a, b_slope, c, faces, polygon)[0]
                   - volume_stats_plane(a, b_slope, c, faces, polygon)[1])
    # find bracket
    lo, hi = -100.0, 100.0
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo * f_hi > 0:
        # no sign change: return mid as fallback
        return (lo + hi) / 2
    c_bal = brentq(fn, lo, hi, xtol=1e-2)
    return c_bal


def compute_batter(pad_poly, a, b_slope, c, params):
    # pad corners with elevation
    corners = list(pad_poly.exterior.coords)[:-1]
    pad_z = [(x, y, a*x + b_slope*y + c) for x, y in corners]
    # offset pad for batter toe
    pad_toe = pad_poly.buffer(params['max_batter_length'], cap_style=2, join_style=2)
    toe_coords = list(pad_toe.exterior.coords)
    # assign toe elevation using max allowed batter slope
    slope = params['batter_max_slope']
    toe_z = []
    for tx, ty in toe_coords:
        # nearest pad corner for drop reference
        d, x0, y0, z0 = min(( (tx-x)**2+(ty-y)**2, x, y, z) for x,y,z in pad_z )
        drop = slope * np.sqrt(d)
        toe_z.append((tx, ty, z0 - drop))
    return pad_z, toe_z


def process_dxf(file_bytes, params):
    # save to temp file
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = tmp.name
    doc = ezdxf.readfile(path)
    centers = extract_pad_centers(doc)
    faces = extract_tin_faces(doc)
    pads = generate_pad_polygons(centers, params['pad_length'], params['pad_width'])

    summary = []
    out = ezdxf.new('AC1032'); msp = out.modelspace()

    for idx, pad in enumerate(pads):
        # collect sample points for plane fit
        samples = [(x,y,z) for tri in faces for x,y,z in tri if pad.contains(Point(x,y))]
        pts = np.array(samples)
        if pts.size == 0:
            continue
        a0, b0, c0 = fit_plane(pts)
        a1, b1 = enforce_slope(a0, b0, params['min_slope'], params['max_slope'])
        c1 = balance_cutfill(a1, b1, faces, pad, params['tolerance'])
        cut, fill = volume_stats_plane(a1, b1, c1, faces, pad)
        summary.append({'pad_index': idx, 'cut': cut, 'fill': fill})
        # draw pad plane
        verts = [(x, y, a1*x + b1*y + c1) for x, y in list(pad.exterior.coords)[:-1]]
        msp.add_3dface(verts)
        # draw batter strings
        pad_z, toe_z = compute_batter(pad, a1, b1, c1, params)
        # connect each pad edge to toe
        for i in range(len(pad_z)):
            j = (i + 1) % len(pad_z)
            pts3d = [pad_z[i], pad_z[j], toe_z[j], toe_z[i]]
            msp.add_polyline3d(pts3d, dxfattribs={'layer': 'BATTER'})

    # export DXF
    buf = io.StringIO()
    out.write(buf)
    dxf_bytes = buf.getvalue().encode('utf-8')
    return dxf_bytes, pd.DataFrame(summary)

# === Streamlit UI ===
st.title("Pad Design & Cut/Fill Balance with Batters")
st.markdown("Upload DXF ('PAD CENTRE POINTS' & 'TIN SURFACE') and adjust parameters.")
up = st.file_uploader("Upload DXF", type=['dxf'])
st.sidebar.header("Design Parameters")
params = {
    'pad_length': st.sidebar.number_input("Pad length (m)", 1.0, 100.0, 10.0),
    'pad_width': st.sidebar.number_input("Pad width (m)", 1.0, 100.0, 10.0),
    'min_slope': st.sidebar.slider("Min pad slope (%)", 0.0, 5.0, 0.5) / 100,
    'max_slope': st.sidebar.slider("Max pad slope (%)", 0.0, 5.0, 2.0) / 100,
    'batter_min_slope': 1 / st.sidebar.slider("Min batter 1:X", 1, 10, 2),
    'batter_max_slope': 1 / st.sidebar.slider("Max batter 1:X", 1, 10, 5),
    'max_batter_length': st.sidebar.number_input("Max batter length (m)", 0.1, 20.0, 4.0),
    'tolerance': st.sidebar.number_input("Cut/Fill tolerance (m³)", 0.0, 100.0, 10.0)
}
if up:
    dxf_out, df = process_dxf(up.read(), params)
    st.download_button("Download Design DXF", dxf_out, file_name="design_with_batters.dxf")
    st.write("### Volume Summary")
    st.dataframe(df)
