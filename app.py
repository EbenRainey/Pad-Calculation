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
    # pts: (3,2) array of XY coords
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))/2


def extract_pad_centers(doc, layer_name="PAD CENTRE POINTS"):
    centers = []
    for pt in doc.modelspace().query(f'POINT[layer=="{layer_name}"]'):
        loc = pt.dxf.location
        centers.append((loc.x, loc.y, getattr(loc, 'z', 0.0)))
    return centers


def extract_tin_faces(doc, layer_name="TIN SURFACE"):
    faces = []
    for face in doc.modelspace().query(f'3DFACE[layer=="{layer_name}"]'):
        # take first three vertices
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
        dx, dy = length/2.0, width/2.0
        corners = [(x-dx, y-dy), (x+dx, y-dy), (x+dx, y+dy), (x-dx, y+dy)]
        pads.append(Polygon(corners))
    return pads


def fit_plane(points):
    A = np.c_[points[:,0], points[:,1], np.ones(len(points))]
    b = points[:,2]
    a, b_slope, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return a, b_slope, c


def enforce_slope(a, b_slope, min_s, max_s):
    grad = np.array([a, b_slope])
    mag = np.linalg.norm(grad)
    target = np.clip(mag, min_s, max_s)
    if mag > 0:
        grad *= target/mag
    return grad[0], grad[1]


def volume_stats_plane(a, b_slope, c, faces, polygon):
    cut, fill = 0.0, 0.0
    for tri in faces:
        # face centroid
        xy = tri[:,:2]
        cx, cy = xy.mean(axis=0)
        if not polygon.contains(Point(cx, cy)):
            continue
        area = triangle_area(xy)
        z_ground = tri[:,2].mean()
        z_plane = a*cx + b_slope*cy + c
        dv = z_plane - z_ground
        if dv>0: cut += dv*area
        else: fill += -dv*area
    return cut, fill


def balance_cutfill(a, b_slope, faces, polygon, tol):
    # find c that balances cut and fill
    fn = lambda c: volume_stats_plane(a,b_slope,c,faces,polygon)[0] - volume_stats_plane(a,b_slope,c,faces,polygon)[1]
    try:
        c_bal = brentq(fn, -100,100,xtol=1)
    except ValueError:
        c_bal = 0.0
    return c_bal


def compute_batter(pad_poly, a, b_slope, c, params):
    # compute pad boundary vertices with elevation
    boundary = list(pad_poly.exterior.coords)[:-1]
    pad_z = [(x,y, a*x + b_slope*y + c) for x,y in boundary]
    # offset pad polygon for batter toe
    pad_toe = pad_poly.buffer(params['max_batter_length'], cap_style=2, join_style=2)
    # approximate toe vertices (match number?) choose exterior coords
    toe_coords = list(pad_toe.exterior.coords)
    # assign toe Z by applying batter slope: drop = slope*distance
    batter_slope = params['batter_min_slope']  # use min (steep) for string
    toe_z = []
    for tx,ty in toe_coords:
        # find nearest pad boundary point
        dists = [((tx-x)**2 + (ty-y)**2, x,y,z) for x,y,z in pad_z]
        d, x0,y0,z0 = min(dists)
        drop = batter_slope * np.sqrt(d)
        toe_z.append((tx,ty, z0 - drop))
    return pad_z, toe_z


def process_dxf(file_bytes, params):
    # write buffer to temp file
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush(); path=tmp.name
    doc = ezdxf.readfile(path)
    centers = extract_pad_centers(doc)
    faces = extract_tin_faces(doc)
    pads = generate_pad_polygons(centers, params['pad_length'], params['pad_width'])

    summary=[]
    out = ezdxf.new('AC1032'); msp=out.modelspace()
    for idx,pad in enumerate(pads):
        # plane fit
        pts=[]
        for tri in faces:
            for x,y,z in tri:
                if pad.contains(Point(x,y)):
                    pts.append((x,y,z))
        pts=np.array(pts)
        if pts.size==0: continue
        a0,b0,c0 = fit_plane(pts)
        a1,b1 = enforce_slope(a0,b0,params['min_slope'],params['max_slope'])
        c1 = balance_cutfill(a1,b1,faces,pad,params['tolerance'])
        cut,fill = volume_stats_plane(a1,b1,c1,faces,pad)
        summary.append({
            'pad_index':idx,'center':centers[idx],'cut':cut,'fill':fill
        })
        # draw pad plane
        v=[]; coords=list(pad.exterior.coords)[:-1]
        for x,y in coords: v.append((x,y,a1*x+b1*y+c1))
        # quad face
        msp.add_3dface([v[0],v[1],v[2],v[3]])
        # batter
        pad_z,toe_z = compute_batter(pad,a1,b1,c1,params)
        # draw batter string
        for i in range(len(toe_z)-1):
            msp.add_lwpolyline([(pad_z[i][0],pad_z[i][1],pad_z[i][2]),toe_z[i],toe_z[i+1],(pad_z[i+1][0],pad_z[i+1][1],pad_z[i+1][2])], dxfattribs={'layer':'BATTER'})
        # close last
        msp.add_lwpolyline([(pad_z[-1][0],pad_z[-1][1],pad_z[-1][2]),toe_z[-1],toe_z[0],(pad_z[0][0],pad_z[0][1],pad_z[0][2])], dxfattribs={'layer':'BATTER'})
    # export
    buf=io.StringIO(); out.write(buf); dxf_bytes=buf.getvalue().encode('utf-8')
    return dxf_bytes, pd.DataFrame(summary)

# === UI ===
st.title("Pad Design & Cut/Fill Balance with Batters")
st.markdown("Upload DXF with 'PAD CENTRE POINTS' & 'TIN SURFACE'.")
up=st.file_uploader("DXF",type=['dxf'])
st.sidebar.header("Params")
params={
 'pad_length':st.sidebar.number_input("Pad L (m)",1.0,100.0,10.0),
 'pad_width':st.sidebar.number_input("Pad W (m)",1.0,100.0,10.0),
 'min_slope':st.sidebar.slider("Min slope%",0.0,5.0,0.5)/100,
 'max_slope':st.sidebar.slider("Max slope%",0.0,5.0,2.0)/100,
 'batter_min_slope':1/st.sidebar.slider("Min batter 1:X",1,10,2),
 'batter_max_slope':1/st.sidebar.slider("Max batter 1:X",1,10,5),
 'max_batter_length':st.sidebar.number_input("Max batter L (m)",0.1,20.0,4.0),
 'tolerance':st.sidebar.number_input("Tol (m³)",0.0,100.0,10.0)
}
if up:
    dxf_out,df=process_dxf(up.read(),params)
    st.download_button("DXF Out",dxf_out,"design_with_batters.dxf")
    st.write(df)
