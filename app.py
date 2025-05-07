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
    return [(pt.dxf.location.x, pt.dxf.location.y)
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
    half_x, half_y = length/2.0, width/2.0
    return [Polygon([(x-half_x, y-half_y), (x+half_x, y-half_y),
                     (x+half_x, y+half_y), (x-half_x, y+half_y)])
            for x, y in centers]


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


def compute_c_limit(corners, uniq, a, b_slope, max_offset):
    c_min, c_max = -np.inf, np.inf
    for x,y in corners:
        dists = np.sum((uniq[:,:2] - np.array([x,y]))**2, axis=1)
        z_ground = uniq[np.argmin(dists),2]
        low = z_ground - a*x - b_slope*y - max_offset
        high = z_ground - a*x - b_slope*y + max_offset
        c_min = max(c_min, low)
        c_max = min(c_max, high)
    return c_min, c_max


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
        if dv > 0:
            cut += dv * area
        else:
            fill += -dv * area
    return cut, fill


def balance_cutfill(a, b_slope, faces, polygon, c_min, c_max):
    fn = lambda cc: (volume_stats_plane(a, b_slope, cc, faces, polygon)[0]
                    - volume_stats_plane(a, b_slope, cc, faces, polygon)[1])
    try:
        f_lo, f_hi = fn(c_min), fn(c_max)
        if f_lo * f_hi > 0:
            return (c_min + c_max) / 2
        return brentq(fn, c_min, c_max, xtol=1e-2)
    except Exception:
        return (c_min + c_max) / 2


def compute_batter(pad_poly, a, b_slope, c, params):
    corners = list(pad_poly.exterior.coords)[:-1]
    subdivisions = params['batter_subdivisions']
    pad_out = []
    toe_out = []
    for i in range(len(corners)):
        x0,y0 = corners[i]
        x1,y1 = corners[(i+1)%len(corners)]
        dx,dy = x1-x0, y1-y0
        # outward normal
        normal = np.array([dy, -dx])
        normal /= np.linalg.norm(normal)
        for k in range(subdivisions+1):
            t = k/subdivisions
            x = x0 + dx*t
            y = y0 + dy*t
            z_pad = a*x + b_slope*y + c
            pad_out.append((x,y,z_pad))
            # toe point
            length = params['max_batter_length']
            x_toe = x + normal[0]*length
            y_toe = y + normal[1]*length
            z_toe = z_pad - params['batter_max_slope']*length
            toe_out.append((x_toe, y_toe, z_toe))
    return pad_out, toe_out


def process_dxf(file_bytes, params):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = tmp.name
    doc = ezdxf.readfile(path)
    centers = extract_pad_centers(doc)
    faces = extract_tin_faces(doc)
    pads = generate_pad_polygons(centers, params['pad_length'], params['pad_width'])

    all_verts = np.vstack(faces).reshape(-1,3)
    uniq = np.unique(all_verts, axis=0)

    summary = []
    out_doc = ezdxf.new('AC1032'); msp = out_doc.modelspace()

    for idx, pad in enumerate(pads):
        pts = np.array([(x,y,z) for tri in faces for x,y,z in tri if pad.contains(Point(x,y))])
        if pts.size == 0:
            continue
        a0,b0,_ = fit_plane(pts)
        a1,b1 = enforce_slope(a0,b0,params['min_slope'],params['max_slope'])
        corners = list(pad.exterior.coords)[:-1]
        c_min,c_max = compute_c_limit(corners, uniq, a1, b1, params['max_offset'])
        c1 = balance_cutfill(a1,b1,faces,pad,c_min,c_max)
        cut,fill = volume_stats_plane(a1,b1,c1,faces,pad)
        summary.append({'pad_index':idx,'cut':cut,'fill':fill})
        # draw pad plane
        verts = [(x,y,a1*x+b1*y+c1) for x,y in corners]
        msp.add_3dface(verts)
        # draw batter string with subdivisions
        pad_pts,toe_pts = compute_batter(pad,a1,b1,c1,params)
        # combine pad then reversed toe for closed polyline
        poly_pts = pad_pts + toe_pts[::-1]
        msp.add_polyline3d(poly_pts, dxfattribs={'layer':'BATTER'})

    buf = io.StringIO()
    out_doc.write(buf)
    dxf_bytes = buf.getvalue().encode('utf-8')
    return dxf_bytes, pd.DataFrame(summary)

# === Streamlit UI ===
st.title("Pad Design & Cut/Fill Balance with Advanced Batters")
st.markdown("Upload DXF ('PAD CENTRE POINTS' & 'TIN SURFACE') and adjust parameters.")
up = st.file_uploader("Upload DXF", type=['dxf'])
st.sidebar.header("Design Parameters")
params = {
    'pad_length': st.sidebar.number_input("Pad length (m)",1.0,100.0,10.0),
    'pad_width': st.sidebar.number_input("Pad width (m)",1.0,100.0,10.0),
    'min_slope': st.sidebar.slider("Min pad slope (%)",0.0,5.0,0.5)/100,
    'max_slope': st.sidebar.slider("Max pad slope (%)",0.0,5.0,2.0)/100,
    'batter_min_slope':1/st.sidebar.slider("Min batter 1:X",1,10,2),
    'batter_max_slope':1/st.sidebar.slider("Max batter 1:X",1,10,5),
    'max_batter_length':st.sidebar.number_input("Max batter length (m)",0.1,20.0,4.0),
    'max_offset':st.sidebar.number_input("Max vertical offset (m)",0.0,10.0,2.0),
    'batter_subdivisions':st.sidebar.slider("Batter subdivisions per edge",1,20,4)
}
if up:
    dxf_out,df = process_dxf(up.read(), params)
    st.download_button("Download Design DXF",dxf_out,file_name="design_with_advanced_batters.dxf")
    st.write("### Volume Summary")
    st.dataframe(df)
