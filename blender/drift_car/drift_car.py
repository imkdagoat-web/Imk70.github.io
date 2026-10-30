"""
IMK70 Drift Car - procedural, ultra-detailed drift car for Blender
===================================================================

Builds a complete wide-body drift coupe plus a photo-real drift scene
(asphalt, tyre marks, tyre smoke, barriers, curbs, golden-hour sky,
camera with depth of field) entirely from code. No external assets.

Works with Blender 4.2 LTS and newer (tested on Blender 5.2).

HOW TO USE
----------
In Blender (GUI):
    1. Open the "Scripting" workspace.
    2. Open this file, press "Run Script".
    3. Press F12 to render (Cycles).

From the command line:
    blender -b -P drift_car.py -- --render drift_car.png --samples 512
    blender -b -P drift_car.py -- --save drift_car.blend

Options (after the "--"):
    --render PATH        render a still to PATH
    --save PATH          save the generated scene as a .blend
    --samples N          Cycles samples (default 384)
    --res WxH            resolution (default 1920x1080)
    --percent P          resolution percentage (default 100)
    --no-smoke           skip the tyre smoke volume (much faster renders)
    --cam NAME           which camera to render: hero, side, rear, top
    --gpu                try to render on the GPU

Everything you might want to tweak (paint colour, wheel colour, steering
angle, drift angle, livery text ...) lives in the CONFIG block below.
"""

import bpy
import bmesh
import math
import os
import random
import sys
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    # body paint (linear RGB) and the colour it shifts to at grazing angles
    "paint": (0.050, 0.010, 0.140),        # midnight purple
    "paint_flip": (0.010, 0.060, 0.170),   # blue pearl flip
    "paint_metallic": 0.65,
    "wheel": (0.30, 0.17, 0.05),           # bronze wheels
    "caliper": (0.80, 0.04, 0.02),         # red calipers
    "cage": (0.85, 0.85, 0.82),            # roll cage colour
    "livery": "IMK70",                     # windshield banner
    "number": "70",                        # door number
    "plate": "IMK 70",                     # rear number plate
    "steer_deg": -24.0,                    # counter-steer (negative = right)
    "drift_deg": 34.0,                     # body angle vs. direction of travel
    "turn_radius": 17.0,                   # metres
    "roll_deg": 1.6,                       # body roll to outside of the turn
    "front_camber": 3.5,                   # degrees negative camber
    "rear_camber": 1.0,
    "smoke": True,
    "seed": 70,
}

# Car dimensions (metres). x = forward, y = left, z = up.
WHEEL_R = 0.315          # 225/40 R18-ish
WHEEL_HALF_W = 0.1225
FRONT_AXLE = 1.30
REAR_AXLE = -1.25
ARCH_R = 0.355
ARCH_Z = 0.33

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], a)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, a, b):
    return max(a, min(b, v))


def link_obj(ob, coll):
    coll.objects.link(ob)
    return ob


def new_collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def bm_to_object(bm, name, coll, mats=(), smooth=True, recalc=True):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def add_subsurf(ob, levels=1, render=2):
    m = ob.modifiers.new("Subdivision", "SUBSURF")
    m.levels = levels
    m.render_levels = render
    return m


def add_bevel(ob, width=0.002, segments=2, angle=35):
    m = ob.modifiers.new("Bevel", "BEVEL")
    m.width = width
    m.segments = segments
    m.limit_method = "ANGLE"
    m.angle_limit = math.radians(angle)
    m.harden_normals = False
    return m


def bake_modifiers(ob):
    """Apply all modifiers without bpy.ops (works in background mode)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    me.name = old.name
    bpy.data.meshes.remove(old)


def make_bvh(ob):
    me = ob.data
    mw = ob.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    return BVHTree.FromPolygons(verts, polys)


def ray(bvh, origin, direction, dist=20.0):
    d = Vector(direction).normalized()
    loc, nor, idx, _ = bvh.ray_cast(Vector(origin), d, dist)
    if loc is None:
        return None, None
    if nor.dot(d) > 0:
        nor = -nor
    return loc, nor


def basis_matrix(x_axis, z_axis, loc):
    z = Vector(z_axis).normalized()
    x = Vector(x_axis)
    x = (x - z * x.dot(z)).normalized()
    y = z.cross(x)
    m = Matrix((
        (x.x, y.x, z.x, loc[0]),
        (x.y, y.y, z.y, loc[1]),
        (x.z, y.z, z.z, loc[2]),
        (0, 0, 0, 1)))
    return m


def catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------


def lathe(name, profile, coll, mats=(), segs=96, mat_fn=None, closed=True,
          a0=0.0, a1=2 * math.pi, smooth=True):
    """Revolve a (radius, y) profile around the local Y axis."""
    bm = bmesh.new()
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    n = len(profile)
    count = segs if full else segs + 1
    rings = []
    for i in range(count):
        a = a0 + (a1 - a0) * i / segs
        ca, sa = math.cos(a), math.sin(a)
        rings.append([bm.verts.new((r * ca, y, r * sa)) for (r, y) in profile])
    m = n if closed else n - 1
    for i in range(segs):
        r0 = rings[i]
        r1 = rings[(i + 1) % count]
        for j in range(m):
            j1 = (j + 1) % n
            f = bm.faces.new((r0[j], r1[j], r1[j1], r0[j1]))
            if mat_fn:
                f.material_index = mat_fn(j)
    if not full and closed:
        for ring in (rings[0], rings[-1]):
            try:
                f = bm.faces.new(ring)
                if mat_fn:
                    f.material_index = mat_fn(0)
            except ValueError:
                pass
    return bm_to_object(bm, name, coll, mats, smooth)


def box_bm(bm, center, size, matrix=None, mat_index=0):
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    vs = []
    for dx, dy, dz in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                       (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)):
        co = Vector((dx * sx, dy * sy, dz * sz))
        if matrix is not None:
            co = matrix.to_3x3() @ co
        vs.append(bm.verts.new((cx + co.x, cy + co.y, cz + co.z)))
    faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    for f in faces:
        bm.faces.new([vs[i] for i in f]).material_index = mat_index
    return vs


def make_box(name, center, size, coll, mats=(), rot=None, bevel=0.0, smooth=False):
    bm = bmesh.new()
    box_bm(bm, (0, 0, 0), size)
    ob = bm_to_object(bm, name, coll, mats, smooth=smooth)
    ob.location = center
    if rot:
        ob.rotation_euler = rot
    if bevel:
        add_bevel(ob, bevel, 3, 60)
    return ob


def cylinder_bm(bm, p0, p1, radius, segs=32, mat_index=0):
    p0, p1 = Vector(p0), Vector(p1)
    axis = (p1 - p0)
    length = axis.length
    rot = axis.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    m = Matrix.Translation((p0 + p1) / 2) @ rot
    res = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs,
                                radius1=radius, radius2=radius, depth=length, matrix=m)
    for v in res["verts"]:
        for f in v.link_faces:
            f.material_index = mat_index
    return res


def tube_curve(name, points, radius, coll, mat, resolution=6, smooth_corners=True):
    """A bent tube following a poly-line (roll cage bars, wiring...)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    if smooth_corners and len(points) > 2:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(points) - 1)
        for bp, p in zip(sp.bezier_points, points):
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        sp.resolution_u = resolution
    else:
        sp = cu.splines.new("POLY")
        sp.points.add(len(points) - 1)
        for pt, p in zip(sp.points, points):
            pt.co = (p[0], p[1], p[2], 1.0)
    cu.materials.append(mat)
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    return ob


# ---------------------------------------------------------------------------
# MATERIALS
# ---------------------------------------------------------------------------
PRINCIPLED = {
    "base": "Base Color", "metal": "Metallic", "rough": "Roughness",
    "coat": "Coat Weight", "coat_rough": "Coat Roughness", "coat_ior": "Coat IOR",
    "trans": "Transmission Weight", "ior": "IOR", "emit": "Emission Color",
    "emit_str": "Emission Strength", "aniso": "Anisotropic", "alpha": "Alpha",
    "spec": "Specular IOR Level", "sheen": "Sheen Weight", "sheen_rough": "Sheen Roughness",
    "aniso_rot": "Anisotropic Rotation",
}


def sset(node, name, val):
    s = node.inputs.get(name)
    if s is not None:
        try:
            s.default_value = val
        except (TypeError, ValueError):
            pass


def new_mat(name):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    return m, nt, out


def principled(nt, **kw):
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    for k, v in kw.items():
        if k == "base" or k == "emit":
            v = rgba(v)
        sset(p, PRINCIPLED[k], v)
    return p


def node(nt, kind, **inputs):
    n = nt.nodes.new(kind)
    for k, v in inputs.items():
        sset(n, k, v)
    return n


def L(nt, a, b):
    nt.links.new(a, b)


def ramp(nt, stops, interp="LINEAR"):
    r = nt.nodes.new("ShaderNodeValToRGB")
    cr = r.color_ramp
    cr.interpolation = interp
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    for el, (pos, col) in zip(cr.elements, stops):
        el.position = pos
        el.color = rgba(col) if len(col) == 3 else col
    return r


def math_node(nt, op, a=None, b=None, clamp_=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp_
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            L(nt, v, n.inputs[i])
    return n.outputs[0]


def obj_coords(nt):
    return nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]


def bump(nt, height, strength=0.2, distance=0.01, normal=None):
    b = nt.nodes.new("ShaderNodeBump")
    sset(b, "Strength", strength)
    sset(b, "Distance", distance)
    L(nt, height, b.inputs["Height"])
    if normal is not None:
        L(nt, normal, b.inputs["Normal"])
    return b.outputs["Normal"]


def finish(nt, out, shader):
    L(nt, shader, out.inputs["Surface"])


def mat_simple(name, col, rough=0.5, metal=0.0, coat=0.0, **kw):
    m, nt, out = new_mat(name)
    p = principled(nt, base=col, rough=rough, metal=metal, coat=coat, **kw)
    finish(nt, out, p.outputs[0])
    m.diffuse_color = rgba(col)
    return m


def mat_paint(name, col, flip, metallic=0.6):
    m, nt, out = new_mat(name)
    p = principled(nt, metal=metallic, rough=0.28, coat=1.0, coat_rough=0.015, coat_ior=1.52)
    lw = node(nt, "ShaderNodeLayerWeight", Blend=0.42)
    r = ramp(nt, [(0.0, col), (0.85, flip), (1.0, flip)])
    L(nt, lw.outputs["Facing"], r.inputs["Fac"])
    L(nt, r.outputs["Color"], p.inputs["Base Color"])
    oc = obj_coords(nt)
    # metallic flakes
    vor = node(nt, "ShaderNodeTexVoronoi", Scale=3200.0)
    L(nt, oc, vor.inputs["Vector"])
    L(nt, bump(nt, vor.outputs["Color"], 0.10, 0.0004), p.inputs["Normal"])
    # orange peel in the clear coat
    nz = node(nt, "ShaderNodeTexNoise", Scale=220.0, Detail=2.0)
    L(nt, oc, nz.inputs["Vector"])
    coat_n = p.inputs.get("Coat Normal")
    if coat_n:
        L(nt, bump(nt, nz.outputs["Fac"], 0.012, 0.002), coat_n)
    # textured black lower valance on the bumpers
    sxb = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, oc, sxb.inputs[0])
    ax = math_node(nt, "ABSOLUTE", sxb.outputs[0])
    zlim = math_node(nt, "ADD", 0.285, math_node(nt, "MULTIPLY", math_node(nt, "LESS_THAN", sxb.outputs[0], 0.0), 0.075))
    m_x = math_node(nt, "GREATER_THAN", ax, 1.98)
    m_z = math_node(nt, "LESS_THAN", sxb.outputs[2], zlim)
    valance = math_node(nt, "MULTIPLY", m_x, m_z)
    tex = node(nt, "ShaderNodeTexNoise", Scale=900.0, Detail=2.0)
    L(nt, oc, tex.inputs["Vector"])
    plastic = principled(nt, base=(0.014, 0.014, 0.015), rough=0.55)
    L(nt, bump(nt, tex.outputs["Fac"], 0.25, 0.002), plastic.inputs["Normal"])
    vmix = nt.nodes.new("ShaderNodeMixShader")
    L(nt, valance, vmix.inputs[0])
    L(nt, p.outputs[0], vmix.inputs[1])
    L(nt, plastic.outputs[0], vmix.inputs[2])
    # inside of the shell (seen through the windows) is dark headliner
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    inner = principled(nt, base=(0.018, 0.018, 0.02), rough=0.85)
    mix = nt.nodes.new("ShaderNodeMixShader")
    L(nt, geo.outputs["Backfacing"], mix.inputs[0])
    L(nt, vmix.outputs[0], mix.inputs[1])
    L(nt, inner.outputs[0], mix.inputs[2])
    finish(nt, out, mix.outputs[0])
    m.diffuse_color = rgba(col)
    return m


def mat_thin_glass(name, tint=(0.6, 0.62, 0.62), rough=0.0):
    """Single-surface glass: transparent + fresnel reflection (car windows)."""
    m, nt, out = new_mat(name)
    tr = node(nt, "ShaderNodeBsdfTransparent", Color=rgba(tint))
    gl = nt.nodes.new("ShaderNodeBsdfGlossy") if hasattr(bpy.types, "ShaderNodeBsdfGlossy") \
        else nt.nodes.new("ShaderNodeBsdfAnisotropic")
    sset(gl, "Roughness", rough)
    fr = node(nt, "ShaderNodeFresnel", IOR=1.52)
    # boost reflectivity a little - laminated glass + coatings
    f2 = math_node(nt, "MULTIPLY_ADD", fr.outputs[0], 1.6)
    f2.node.inputs[2].default_value = 0.02
    f2.node.use_clamp = True
    mix = nt.nodes.new("ShaderNodeMixShader")
    L(nt, f2, mix.inputs[0])
    L(nt, tr.outputs[0], mix.inputs[1])
    L(nt, gl.outputs[0], mix.inputs[2])
    finish(nt, out, mix.outputs[0])
    m.diffuse_color = rgba(tint, 0.3)
    return m


def mat_solid_glass(name, tint=(1, 1, 1), rough=0.0):
    m, nt, out = new_mat(name)
    p = principled(nt, base=tint, rough=rough, trans=1.0, ior=1.5)
    finish(nt, out, p.outputs[0])
    return m


def mat_emission(name, col, strength):
    m, nt, out = new_mat(name)
    e = node(nt, "ShaderNodeEmission", Color=rgba(col), Strength=strength)
    finish(nt, out, e.outputs[0])
    m.diffuse_color = rgba(col)
    return m


def mat_carbon(name):
    m, nt, out = new_mat(name)
    oc = obj_coords(nt)
    mp = node(nt, "ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(45))
    mp.inputs["Scale"].default_value = (160, 160, 160)
    L(nt, oc, mp.inputs["Vector"])
    # twill weave: two sets of stripes, checkered
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, mp.outputs[0], sx.inputs[0])
    chk = node(nt, "ShaderNodeTexChecker", Scale=1.0)
    L(nt, mp.outputs[0], chk.inputs["Vector"])
    wx = math_node(nt, "SINE", math_node(nt, "MULTIPLY", sx.outputs[0], 6.2832 * 2))
    wy = math_node(nt, "SINE", math_node(nt, "MULTIPLY", sx.outputs[1], 6.2832 * 2))
    a = math_node(nt, "ABSOLUTE", wx)
    b = math_node(nt, "ABSOLUTE", wy)
    weave = nt.nodes.new("ShaderNodeMix")
    weave.data_type = "FLOAT"
    L(nt, chk.outputs["Fac"], weave.inputs[0])
    L(nt, a, weave.inputs[2])
    L(nt, b, weave.inputs[3])
    wv = weave.outputs[0]
    col = ramp(nt, [(0.0, (0.004, 0.004, 0.005)), (1.0, (0.035, 0.036, 0.04))])
    L(nt, wv, col.inputs["Fac"])
    p = principled(nt, rough=0.32, coat=1.0, coat_rough=0.02, aniso=0.5)
    L(nt, col.outputs["Color"], p.inputs["Base Color"])
    L(nt, bump(nt, wv, 0.25, 0.001), p.inputs["Normal"])
    finish(nt, out, p.outputs[0])
    m.diffuse_color = (0.02, 0.02, 0.022, 1)
    return m


def mat_tire(name):
    m, nt, out = new_mat(name)
    oc = obj_coords(nt)
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, oc, sx.inputs[0])
    x, y, z = sx.outputs[0], sx.outputs[1], sx.outputs[2]
    ang = math_node(nt, "ARCTAN2", z, x)
    rad = math_node(nt, "SQRT", math_node(nt, "ADD", math_node(nt, "MULTIPLY", x, x),
                                          math_node(nt, "MULTIPLY", z, z)))
    # directional (chevron) tread blocks
    lat = math_node(nt, "MULTIPLY", math_node(nt, "ABSOLUTE", y), 22.0)
    phase = math_node(nt, "ADD", math_node(nt, "MULTIPLY", ang, 34.0), lat)
    sipes = math_node(nt, "SINE", phase)
    grooves = ramp(nt, [(0.0, (0, 0, 0)), (0.86, (0, 0, 0)), (0.93, (1, 1, 1))])
    L(nt, sipes, grooves.inputs["Fac"])
    tread_mask = math_node(nt, "MINIMUM", 1.0, math_node(nt, "MULTIPLY",
                           math_node(nt, "SUBTRACT", rad, 0.306), 400.0))
    tm = math_node(nt, "MAXIMUM", tread_mask, 0.0)
    h = math_node(nt, "MULTIPLY", grooves.outputs[0], tm)
    h = math_node(nt, "MULTIPLY", h, -1.0)
    # sidewall: fine concentric ribs + raised lettering band
    ribs = math_node(nt, "SINE", math_node(nt, "MULTIPLY", rad, 2600.0))
    side_mask = math_node(nt, "SUBTRACT", 1.0, tm)
    h = math_node(nt, "ADD", h, math_node(nt, "MULTIPLY", ribs, math_node(nt, "MULTIPLY", side_mask, 0.04)))
    # wear: lighter scrubbed rubber on the tread, dusty sidewalls
    nz = node(nt, "ShaderNodeTexNoise", Scale=60.0, Detail=8.0)
    L(nt, oc, nz.inputs["Vector"])
    wear = ramp(nt, [(0.35, (0.011, 0.011, 0.012)), (0.75, (0.026, 0.025, 0.025))])
    L(nt, nz.outputs["Fac"], wear.inputs["Fac"])
    p = principled(nt, rough=0.62, spec=0.35)
    L(nt, wear.outputs["Color"], p.inputs["Base Color"])
    rg = ramp(nt, [(0.0, (0.72, 0.72, 0.72)), (1.0, (0.52, 0.52, 0.52))])
    L(nt, tm, rg.inputs["Fac"])
    L(nt, rg.outputs["Color"], p.inputs["Roughness"])
    L(nt, bump(nt, h, 0.9, 0.006), p.inputs["Normal"])
    finish(nt, out, p.outputs[0])
    m.diffuse_color = (0.02, 0.02, 0.02, 1)
    return m


def mat_brake_disc(name):
    m, nt, out = new_mat(name)
    oc = obj_coords(nt)
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, oc, sx.inputs[0])
    x, z = sx.outputs[0], sx.outputs[2]
    rad = math_node(nt, "SQRT", math_node(nt, "ADD", math_node(nt, "MULTIPLY", x, x),
                                          math_node(nt, "MULTIPLY", z, z)))
    ang = math_node(nt, "ARCTAN2", z, x)
    # curved slots
    slot = math_node(nt, "SINE", math_node(nt, "ADD", math_node(nt, "MULTIPLY", ang, 12.0),
                                           math_node(nt, "MULTIPLY", rad, 40.0)))
    sl = ramp(nt, [(0.0, (0, 0, 0)), (0.93, (0, 0, 0)), (0.97, (1, 1, 1))])
    L(nt, slot, sl.inputs["Fac"])
    heat = ramp(nt, [(0.0, (0.42, 0.42, 0.44)), (0.6, (0.45, 0.42, 0.38)),
                     (0.8, (0.36, 0.30, 0.38)), (1.0, (0.30, 0.34, 0.45))])
    L(nt, math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", rad, 0.10), 13.0), heat.inputs["Fac"])
    p = principled(nt, metal=1.0, rough=0.38, aniso=0.8)
    L(nt, heat.outputs["Color"], p.inputs["Base Color"])
    L(nt, bump(nt, math_node(nt, "MULTIPLY", sl.outputs[0], -1.0), 0.8, 0.004), p.inputs["Normal"])
    finish(nt, out, p.outputs[0])
    return m


def mat_titanium(name):
    m, nt, out = new_mat(name)
    oc = obj_coords(nt)
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, oc, sx.inputs[0])
    r = ramp(nt, [(0.0, (0.55, 0.52, 0.48)), (0.3, (0.60, 0.45, 0.18)),
                  (0.55, (0.35, 0.15, 0.40)), (0.8, (0.12, 0.22, 0.55)), (1.0, (0.45, 0.45, 0.50))])
    fac = math_node(nt, "MULTIPLY_ADD", sx.outputs[1], -4.0)
    fac.node.inputs[2].default_value = 0.55
    L(nt, fac, r.inputs["Fac"])
    p = principled(nt, metal=1.0, rough=0.22, aniso=0.6)
    L(nt, r.outputs["Color"], p.inputs["Base Color"])
    finish(nt, out, p.outputs[0])
    return m


def mat_asphalt(name, centre=None, r_in=0.0, r_out=0.0):
    """Track asphalt. With `centre`, only the ring r_in..r_out is asphalt, the rest is dry grass."""
    m, nt, out = new_mat(name)
    oc = obj_coords(nt)
    big = node(nt, "ShaderNodeTexNoise", Scale=0.35, Detail=6.0, Roughness=0.6)
    mid = node(nt, "ShaderNodeTexNoise", Scale=6.0, Detail=10.0, Roughness=0.65)
    agg = node(nt, "ShaderNodeTexVoronoi", Scale=650.0)
    fine = node(nt, "ShaderNodeTexNoise", Scale=900.0, Detail=2.0)
    for t in (big, mid, agg, fine):
        L(nt, oc, t.inputs["Vector"])
    base = ramp(nt, [(0.3, (0.030, 0.030, 0.031)), (0.7, (0.085, 0.082, 0.078))])
    L(nt, mid.outputs["Fac"], base.inputs["Fac"])
    stones = ramp(nt, [(0.0, (1, 1, 1)), (0.08, (0, 0, 0))])
    L(nt, agg.outputs["Distance"], stones.inputs["Fac"])
    col = nt.nodes.new("ShaderNodeMix")
    col.data_type = "RGBA"
    col.blend_type = "SCREEN"
    L(nt, math_node(nt, "MULTIPLY", stones.outputs[0], 0.35), col.inputs[0])
    L(nt, base.outputs["Color"], col.inputs[6])
    col.inputs[7].default_value = (0.24, 0.23, 0.22, 1)
    # large-scale patches of rubber / repairs
    patches = ramp(nt, [(0.42, (1, 1, 1)), (0.62, (0.55, 0.55, 0.55))])
    L(nt, big.outputs["Fac"], patches.inputs["Fac"])
    col2 = nt.nodes.new("ShaderNodeMix")
    col2.data_type = "RGBA"
    col2.blend_type = "MULTIPLY"
    col2.inputs[0].default_value = 1.0
    L(nt, col.outputs[2], col2.inputs[6])
    L(nt, patches.outputs["Color"], col2.inputs[7])
    p = principled(nt, rough=0.82)
    L(nt, col2.outputs[2], p.inputs["Base Color"])
    rr = ramp(nt, [(0.0, (0.95, 0.95, 0.95)), (1.0, (0.62, 0.62, 0.62))])
    L(nt, big.outputs["Fac"], rr.inputs["Fac"])
    L(nt, rr.outputs["Color"], p.inputs["Roughness"])
    h = math_node(nt, "ADD", math_node(nt, "MULTIPLY", agg.outputs["Distance"], 1.0),
                  math_node(nt, "MULTIPLY", fine.outputs["Fac"], 0.5))
    L(nt, bump(nt, h, 0.45, 0.004), p.inputs["Normal"])
    if centre is None:
        finish(nt, out, p.outputs[0])
        return m
    # dry grass outside the track
    gn = node(nt, "ShaderNodeTexNoise", Scale=1.5, Detail=8.0)
    gf = node(nt, "ShaderNodeTexNoise", Scale=160.0, Detail=4.0)
    L(nt, oc, gn.inputs["Vector"])
    L(nt, oc, gf.inputs["Vector"])
    gc = ramp(nt, [(0.3, (0.045, 0.070, 0.015)), (0.55, (0.090, 0.095, 0.030)), (0.75, (0.15, 0.13, 0.055))])
    L(nt, gn.outputs["Fac"], gc.inputs["Fac"])
    g = principled(nt, rough=0.95, spec=0.12)
    L(nt, gc.outputs["Color"], g.inputs["Base Color"])
    L(nt, bump(nt, gf.outputs["Fac"], 0.8, 0.02), g.inputs["Normal"])
    vm = nt.nodes.new("ShaderNodeVectorMath")
    vm.operation = "DISTANCE"
    L(nt, oc, vm.inputs[0])
    vm.inputs[1].default_value = (centre.x, centre.y, 0.0)
    edge = math_node(nt, "MULTIPLY", gn.outputs["Fac"], 0.6)
    r = math_node(nt, "ADD", vm.outputs["Value"], edge)
    inner = nt.nodes.new("ShaderNodeMapRange")
    inner.interpolation_type = "SMOOTHSTEP"
    L(nt, r, inner.inputs["Value"])
    inner.inputs["From Min"].default_value = r_in - 0.2
    inner.inputs["From Max"].default_value = r_in + 0.2
    outer = nt.nodes.new("ShaderNodeMapRange")
    outer.interpolation_type = "SMOOTHSTEP"
    L(nt, r, outer.inputs["Value"])
    outer.inputs["From Min"].default_value = r_out - 0.2
    outer.inputs["From Max"].default_value = r_out + 0.2
    on_track = math_node(nt, "MULTIPLY", inner.outputs[0], math_node(nt, "SUBTRACT", 1.0, outer.outputs[0]))
    mix = nt.nodes.new("ShaderNodeMixShader")
    L(nt, on_track, mix.inputs[0])
    L(nt, g.outputs[0], mix.inputs[1])
    L(nt, p.outputs[0], mix.inputs[2])
    finish(nt, out, mix.outputs[0])
    return m


def mat_chainlink(name):
    m, nt, out = new_mat(name)
    fu = node(nt, "ShaderNodeAttribute")
    fu.attribute_name = "fence_u"
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, obj_coords(nt), sx.inputs[0])
    u, v = fu.outputs["Fac"], sx.outputs[2]
    k = 2 * math.pi / 0.06
    a = math_node(nt, "SINE", math_node(nt, "MULTIPLY", math_node(nt, "ADD", u, v), k))
    b = math_node(nt, "SINE", math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", u, v), k))
    # wires sit on the zero crossings -> ~10% coverage like real chain-link
    wire = math_node(nt, "MINIMUM", math_node(nt, "ABSOLUTE", a), math_node(nt, "ABSOLUTE", b))
    w = ramp(nt, [(0.0, (1, 1, 1)), (0.018, (1, 1, 1)), (0.032, (0, 0, 0))])
    L(nt, wire, w.inputs["Fac"])
    p = principled(nt, base=(0.22, 0.22, 0.21), metal=0.6, rough=0.6)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    L(nt, w.outputs[0], mix.inputs[0])
    L(nt, tr.outputs[0], mix.inputs[1])
    L(nt, p.outputs[0], mix.inputs[2])
    finish(nt, out, mix.outputs[0])
    return m


def mat_foliage(name):
    m, nt, out = new_mat(name)
    nz = node(nt, "ShaderNodeTexNoise", Scale=3.0, Detail=6.0)
    L(nt, obj_coords(nt), nz.inputs["Vector"])
    c = ramp(nt, [(0.3, (0.012, 0.030, 0.008)), (0.7, (0.050, 0.075, 0.020))])
    L(nt, nz.outputs["Fac"], c.inputs["Fac"])
    p = principled(nt, rough=0.85, spec=0.15)
    L(nt, c.outputs["Color"], p.inputs["Base Color"])
    vo = node(nt, "ShaderNodeTexVoronoi", Scale=14.0)
    L(nt, obj_coords(nt), vo.inputs["Vector"])
    L(nt, bump(nt, vo.outputs["Distance"], 1.0, 0.2), p.inputs["Normal"])
    finish(nt, out, p.outputs[0])
    return m


def mat_concrete(name):
    m, nt, out = new_mat(name)
    oc = obj_coords(nt)
    nz = node(nt, "ShaderNodeTexNoise", Scale=4.0, Detail=12.0)
    fn = node(nt, "ShaderNodeTexNoise", Scale=180.0, Detail=4.0)
    L(nt, oc, nz.inputs["Vector"])
    L(nt, oc, fn.inputs["Vector"])
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, oc, sx.inputs[0])
    col = ramp(nt, [(0.3, (0.32, 0.31, 0.29)), (0.7, (0.52, 0.50, 0.47))])
    L(nt, nz.outputs["Fac"], col.inputs["Fac"])
    # rubber scuffs from cars kissing the wall
    scuff = node(nt, "ShaderNodeTexNoise", Scale=1.2, Detail=6.0)
    L(nt, oc, scuff.inputs["Vector"])
    sc = ramp(nt, [(0.55, (1, 1, 1)), (0.68, (0.08, 0.08, 0.08))])
    L(nt, scuff.outputs["Fac"], sc.inputs["Fac"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    L(nt, col.outputs[0], mix.inputs[6])
    L(nt, sc.outputs[0], mix.inputs[7])
    p = principled(nt, rough=0.9)
    L(nt, mix.outputs[2], p.inputs["Base Color"])
    L(nt, bump(nt, fn.outputs["Fac"], 0.3, 0.003), p.inputs["Normal"])
    finish(nt, out, p.outputs[0])
    return m


def mat_curb(name):
    m, nt, out = new_mat(name)
    attr = node(nt, "ShaderNodeAttribute")
    attr.attribute_name = "curb_u"
    s = math_node(nt, "FRACT", math_node(nt, "MULTIPLY", attr.outputs["Fac"], 0.5))
    r = ramp(nt, [(0.0, (0.55, 0.02, 0.015)), (0.5, (0.75, 0.74, 0.72))], "CONSTANT")
    L(nt, s, r.inputs["Fac"])
    nz = node(nt, "ShaderNodeTexNoise", Scale=30.0, Detail=8.0)
    L(nt, obj_coords(nt), nz.inputs["Vector"])
    dirt = ramp(nt, [(0.3, (1, 1, 1)), (0.8, (0.35, 0.33, 0.32))])
    L(nt, nz.outputs["Fac"], dirt.inputs["Fac"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    L(nt, r.outputs[0], mix.inputs[6])
    L(nt, dirt.outputs[0], mix.inputs[7])
    p = principled(nt, rough=0.55)
    L(nt, mix.outputs[2], p.inputs["Base Color"])
    finish(nt, out, p.outputs[0])
    return m


def mat_rubber_marks(name, strength=0.85):
    m, nt, out = new_mat(name)
    attr = node(nt, "ShaderNodeAttribute")
    attr.attribute_name = "mark"
    oc = obj_coords(nt)
    nz = node(nt, "ShaderNodeTexNoise", Scale=3.0, Detail=10.0)
    st = node(nt, "ShaderNodeTexWave", Scale=60.0, Distortion=4.0)
    L(nt, oc, nz.inputs["Vector"])
    L(nt, oc, st.inputs["Vector"])
    a = ramp(nt, [(0.25, (0, 0, 0)), (0.65, (1, 1, 1))])
    L(nt, nz.outputs["Fac"], a.inputs["Fac"])
    alpha = math_node(nt, "MULTIPLY", a.outputs[0], attr.outputs["Fac"])
    alpha = math_node(nt, "MULTIPLY", alpha, math_node(nt, "ADD", math_node(nt, "MULTIPLY", st.outputs["Fac"], 0.35), 0.65))
    alpha = math_node(nt, "MULTIPLY", alpha, strength, clamp_=True)
    p = principled(nt, base=(0.008, 0.008, 0.008), rough=0.5)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    L(nt, alpha, mix.inputs[0])
    L(nt, tr.outputs[0], mix.inputs[1])
    L(nt, p.outputs[0], mix.inputs[2])
    finish(nt, out, mix.outputs[0])
    return m


def mat_smoke(name):
    m, nt, out = new_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    gen = tc.outputs["Object"]
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(nt, gen, sx.inputs[0])
    x, y, z = sx.outputs[0], sx.outputs[1], sx.outputs[2]
    # ellipsoidal falloff
    d2 = math_node(nt, "ADD", math_node(nt, "ADD", math_node(nt, "MULTIPLY", x, x),
                                        math_node(nt, "MULTIPLY", y, y)),
                   math_node(nt, "MULTIPLY", z, z))
    fall = math_node(nt, "SUBTRACT", 1.0, math_node(nt, "SQRT", d2), clamp_=True)
    fall = math_node(nt, "POWER", fall, 0.9)
    # thicker near the tyres (+x end) and near the ground (-z)
    along = math_node(nt, "MULTIPLY_ADD", x, 0.35)
    along.node.inputs[2].default_value = 0.65
    low = math_node(nt, "MULTIPLY_ADD", z, -0.35)
    low.node.inputs[2].default_value = 0.8
    shape = math_node(nt, "MULTIPLY", math_node(nt, "MULTIPLY", fall, along), low)
    # billowing turbulence
    mp = node(nt, "ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.6, 1.6)
    L(nt, gen, mp.inputs["Vector"])
    warp = node(nt, "ShaderNodeTexNoise", Scale=1.4, Detail=4.0)
    L(nt, mp.outputs[0], warp.inputs["Vector"])
    vadd = nt.nodes.new("ShaderNodeVectorMath")
    vadd.operation = "ADD"
    L(nt, mp.outputs[0], vadd.inputs[0])
    L(nt, warp.outputs["Color"], vadd.inputs[1])
    nz = node(nt, "ShaderNodeTexNoise", Scale=3.2, Detail=12.0, Roughness=0.68)
    L(nt, vadd.outputs[0], nz.inputs["Vector"])
    thr = ramp(nt, [(0.45, (0, 0, 0)), (0.62, (1, 1, 1))])
    L(nt, nz.outputs["Fac"], thr.inputs["Fac"])
    dens = math_node(nt, "MULTIPLY", math_node(nt, "MULTIPLY", thr.outputs[0], shape), 9.0)
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    sset(vol, "Color", (0.95, 0.95, 0.96, 1))
    sset(vol, "Anisotropy", 0.25)
    L(nt, dens, vol.inputs["Density"])
    # cheap stand-in for multiple scattering: dense smoke glows softly
    sset(vol, "Emission Color", (1.0, 0.93, 0.85, 1))
    L(nt, math_node(nt, "MULTIPLY", dens, 0.045), vol.inputs["Emission Strength"])
    L(nt, vol.outputs[0], out.inputs["Volume"])
    return m


def mat_vinyl(name, col, rough=0.35):
    return mat_simple(name, col, rough=rough, coat=0.4)


def build_materials():
    C = CONFIG
    M = {}
    M["paint"] = mat_paint("Paint", C["paint"], C["paint_flip"], C["paint_metallic"])
    M["glass"] = mat_thin_glass("Glass Windshield", (0.62, 0.66, 0.64))
    M["glass_tint"] = mat_thin_glass("Glass Tinted", (0.20, 0.21, 0.22))
    M["trim"] = mat_simple("Gloss Black Trim", (0.008, 0.008, 0.009), rough=0.12, coat=1.0)
    M["satin_black"] = mat_simple("Satin Black", (0.015, 0.015, 0.016), rough=0.45)
    M["liner"] = mat_simple("Arch Liner", (0.012, 0.012, 0.013), rough=0.85)
    M["gap"] = mat_simple("Panel Gap", (0.004, 0.004, 0.004), rough=0.9)
    M["under"] = mat_simple("Underbody", (0.025, 0.025, 0.027), rough=0.7)
    M["housing"] = mat_simple("Black Chrome", (0.06, 0.06, 0.065), rough=0.12, metal=1.0)
    M["chrome"] = mat_simple("Chrome", (0.92, 0.92, 0.93), rough=0.04, metal=1.0)
    M["alu"] = mat_simple("Polished Aluminium", (0.91, 0.91, 0.92), rough=0.10, metal=1.0, aniso=0.6)
    M["alu_brushed"] = mat_simple("Brushed Aluminium", (0.80, 0.80, 0.82), rough=0.28, metal=1.0, aniso=0.85)
    M["anodized"] = mat_simple("Anodized Black", (0.03, 0.03, 0.035), rough=0.3, metal=1.0)
    M["wheel"] = mat_simple("Wheel Paint", C["wheel"], rough=0.24, metal=1.0, coat=0.6)
    M["caliper"] = mat_simple("Caliper Paint", C["caliper"], rough=0.25, coat=1.0)
    M["tire"] = mat_tire("Tyre Rubber")
    M["disc"] = mat_brake_disc("Brake Disc")
    M["carbon"] = mat_carbon("Carbon Fibre")
    M["titanium"] = mat_titanium("Burnt Titanium")
    M["soot"] = mat_simple("Soot", (0.01, 0.009, 0.008), rough=0.95)
    M["lens"] = mat_thin_glass("Lens Clear", (0.97, 0.97, 0.97))
    M["lens_red"] = mat_thin_glass("Lens Red", (0.75, 0.04, 0.03))
    M["lens_smoke"] = mat_thin_glass("Lens Smoked", (0.10, 0.08, 0.08))
    M["projector"] = mat_solid_glass("Projector Glass")
    M["led_white"] = mat_emission("LED White", (1.0, 0.97, 0.92), 28.0)
    M["led_red"] = mat_emission("LED Red", (1.0, 0.03, 0.01), 14.0)
    M["led_amber"] = mat_emission("LED Amber", (1.0, 0.45, 0.02), 4.0)
    M["intercooler"] = mat_simple("Intercooler", (0.55, 0.55, 0.57), rough=0.35, metal=1.0)
    M["fabric"] = mat_simple("Seat Fabric", (0.02, 0.02, 0.022), rough=0.95, sheen=0.6)
    M["fabric_red"] = mat_simple("Seat Fabric Red", (0.35, 0.01, 0.01), rough=0.95, sheen=0.6)
    M["suede"] = mat_simple("Alcantara", (0.025, 0.025, 0.025), rough=1.0, sheen=1.0)
    M["cage"] = mat_simple("Cage Paint", C["cage"], rough=0.38, coat=0.3)
    M["harness"] = mat_simple("Harness", (0.45, 0.02, 0.02), rough=0.8, sheen=0.5)
    M["dash"] = mat_simple("Dash Plastic", (0.02, 0.02, 0.021), rough=0.7)
    M["vinyl_white"] = mat_vinyl("Vinyl White", (0.85, 0.85, 0.84))
    M["vinyl_black"] = mat_vinyl("Vinyl Black", (0.01, 0.01, 0.01))
    M["vinyl_accent"] = mat_vinyl("Vinyl Accent", (0.9, 0.35, 0.02))
    M["plate"] = mat_simple("Plate", (0.85, 0.85, 0.80), rough=0.3, coat=0.5)
    M["strap"] = mat_simple("Tow Strap", (0.7, 0.02, 0.01), rough=0.8, sheen=0.4)
    _, _, centre = drift_geometry()
    R = CONFIG["turn_radius"]
    M["asphalt"] = mat_asphalt("Asphalt", centre, R - 7.6, R + 9.7)
    M["chainlink"] = mat_chainlink("Chain Link Fence")
    M["foliage"] = mat_foliage("Foliage")
    M["galvanised"] = mat_simple("Galvanised Steel", (0.55, 0.56, 0.56), rough=0.45, metal=1.0)
    M["concrete"] = mat_concrete("Concrete")
    M["curb"] = mat_curb("Curb")
    M["marks"] = mat_rubber_marks("Tyre Marks")
    M["marks_light"] = mat_rubber_marks("Tyre Marks Light", 0.35)
    M["smoke"] = mat_smoke("Tyre Smoke")
    M["tyre_wall"] = mat_simple("Old Tyres", (0.02, 0.02, 0.02), rough=0.8)
    M["paint_line"] = mat_simple("Line Paint", (0.70, 0.70, 0.66), rough=0.6)
    return M


# ---------------------------------------------------------------------------
# BODY SHELL
# ---------------------------------------------------------------------------
# key cross sections, front to rear:
#   x, z_bottom, z_top(centre), half_width, z_of_max_width, z_beltline, top_half_width
BODY_KEYS = [
    (2.215, 0.265, 0.610, 0.660, 0.40, 0.555, 0.500),
    (2.145, 0.205, 0.680, 0.825, 0.42, 0.620, 0.680),
    (1.970, 0.170, 0.740, 0.888, 0.48, 0.680, 0.760),
    (1.620, 0.155, 0.795, 0.905, 0.52, 0.735, 0.790),
    (1.300, 0.150, 0.820, 0.915, 0.52, 0.760, 0.790),
    (0.950, 0.150, 0.860, 0.890, 0.52, 0.800, 0.750),
    (0.550, 0.150, 1.050, 0.875, 0.52, 0.830, 0.660),
    (0.150, 0.150, 1.240, 0.875, 0.52, 0.850, 0.600),
    (-0.400, 0.150, 1.265, 0.885, 0.53, 0.860, 0.600),
    (-0.850, 0.150, 1.220, 0.905, 0.53, 0.870, 0.610),
    (-1.250, 0.155, 1.080, 0.930, 0.55, 0.890, 0.700),
    (-1.600, 0.165, 0.960, 0.925, 0.56, 0.900, 0.790),
    (-2.000, 0.190, 0.975, 0.885, 0.57, 0.920, 0.790),
    (-2.180, 0.240, 0.950, 0.820, 0.58, 0.900, 0.720),
    (-2.270, 0.310, 0.880, 0.680, 0.58, 0.830, 0.560),
]

WINDSHIELD = (0.17, 0.93)
REAR_WINDOW = (-1.56, -0.87)
SIDE_WINDOW = (-1.06, 0.88)
B_PILLAR = (-0.47, -0.39)


def ring_half(st):
    x, zb, zt, ws, zw, zbelt, wt = st
    p5z = zbelt + 0.015
    p6z = max(zt - 0.045, p5z + 0.012)
    return [
        (0.0, zb),
        (ws - 0.10, zb),
        (ws - 0.025, zb + 0.05),
        (ws, zw),
        (ws - 0.02, zbelt),
        (ws - 0.075, p5z),
        (wt, p6z),
        (wt - 0.06, zt - 0.02),
        (wt * 0.45, zt - 0.004),
        (0.0, zt),
    ]


def full_ring(half):
    pts = [(y, z) for (y, z) in half]
    pts += [(-y, z) for (y, z) in reversed(half[1:-1])]
    return pts  # 18 points


def half_index(k):
    return k if k <= 9 else 18 - k


def sample_stations():
    keys = BODY_KEYS
    out = []
    n = len(keys)
    for i in range(n - 1):
        p0 = keys[max(i - 1, 0)]
        p1 = keys[i]
        p2 = keys[i + 1]
        p3 = keys[min(i + 2, n - 1)]
        seg = abs(p1[0] - p2[0])
        steps = max(2, int(round(seg / 0.075)))
        for s in range(steps):
            t = s / steps
            out.append(tuple(catmull(p0[k], p1[k], p2[k], p3[k], t) for k in range(7)))
    out.append(keys[-1])
    return out


def band_material(band, x):
    """0 paint, 1 windshield glass, 2 trim, 3 under, 4 tinted glass"""
    in_cabin = REAR_WINDOW[0] - 0.02 < x < WINDSHIELD[1] + 0.02
    if band == 0:
        return 3
    if band == 5:
        if SIDE_WINDOW[0] < x < SIDE_WINDOW[1]:
            if B_PILLAR[0] < x < B_PILLAR[1]:
                return 2
            return 4
        if in_cabin:
            return 2
        return 0
    if band == 6:
        if WINDSHIELD[0] - 0.03 < x < WINDSHIELD[1] or REAR_WINDOW[0] < x < REAR_WINDOW[1] + 0.03:
            return 2
        return 0
    if band in (7, 8):
        if WINDSHIELD[0] < x < WINDSHIELD[1]:
            return 1
        if REAR_WINDOW[0] < x < REAR_WINDOW[1]:
            return 4
    return 0


def build_body_shell(coll, M):
    stations = sample_stations()
    bm = bmesh.new()
    crease = bm.edges.layers.float.get("crease_edge") or bm.edges.layers.float.new("crease_edge")
    rings = []
    for st in stations:
        x = st[0]
        rings.append([bm.verts.new((x, y, z)) for (y, z) in full_ring(ring_half(st))])
    nring = len(rings[0])

    def end_ring(base_ring, dx, scale):
        cz = sum(v.co.z for v in base_ring) / nring
        return [bm.verts.new((v.co.x + dx, v.co.y * scale, cz + (v.co.z - cz) * scale)) for v in base_ring]

    front_cap = end_ring(rings[0], 0.025, 0.62)
    rear_cap = end_ring(rings[-1], -0.025, 0.62)
    all_rings = [front_cap] + rings + [rear_cap]

    for i in range(len(all_rings) - 1):
        ra, rb = all_rings[i], all_rings[i + 1]
        xc = (ra[0].co.x + rb[0].co.x) / 2
        for j in range(nring):
            j1 = (j + 1) % nring
            f = bm.faces.new((ra[j], ra[j1], rb[j1], rb[j]))
            band = min(half_index(j), half_index(j1))
            if half_index(j) == 0 and half_index(j1) == 1:
                band = 0
            f.material_index = band_material(band, xc)
            # crisp shoulder / character lines
            if 0 < i < len(all_rings) - 2:
                for k in (j,):
                    hk = half_index(k)
                    if hk in (3, 4, 6):
                        e = bm.edges.get((ra[k], rb[k]))
                        if e:
                            e[crease] = {3: 0.35, 4: 0.75, 6: 0.45}[hk]
    for cap in (front_cap, rear_cap):
        f = bm.faces.new(cap)
        f.material_index = 0
    ob = bm_to_object(bm, "Body", coll, [M["paint"], M["glass"], M["trim"], M["under"], M["glass_tint"]])
    add_subsurf(ob, 3, 3)
    bake_modifiers(ob)
    return ob


# ---------------------------------------------------------------------------
# surface patches (lights, vents, grilles)
# ---------------------------------------------------------------------------


def patch_grid(bvh, corners, direction, nu=14, nv=8, back=2.0):
    """Project a bilinear grid onto the body. corners = A(u0v0) B(u1v0) C(u1v1) D(u0v1)."""
    d = Vector(direction).normalized()
    A, B, Cc, D = [Vector(c) for c in corners]
    grid = []
    for j in range(nv + 1):
        v = j / nv
        row = []
        for i in range(nu + 1):
            u = i / nu
            p = (A * (1 - u) + B * u) * (1 - v) + (D * (1 - u) + Cc * u) * v
            loc, nor = ray(bvh, p - d * back, d, back * 3)
            row.append((loc, nor, p))
        grid.append(row)
    # fill misses from neighbours
    for j in range(nv + 1):
        for i in range(nu + 1):
            if grid[j][i][0] is None:
                for di in range(1, nu + 1):
                    for ii in (i - di, i + di):
                        if 0 <= ii <= nu and grid[j][ii][0] is not None:
                            loc, nor, p = grid[j][ii]
                            grid[j][i] = (loc + (grid[j][i][2] - p), nor, grid[j][i][2])
                            break
                    if grid[j][i][0] is not None:
                        break
    return grid


def grid_point(grid, u, v):
    nv = len(grid) - 1
    nu = len(grid[0]) - 1
    i = clamp(int(u * nu), 0, nu - 1)
    j = clamp(int(v * nv), 0, nv - 1)
    fu = u * nu - i
    fv = v * nv - j
    p00, p10 = grid[j][i][0], grid[j][i + 1][0]
    p01, p11 = grid[j + 1][i][0], grid[j + 1][i + 1][0]
    n = grid[j][i][1]
    return (p00 * (1 - fu) + p10 * fu) * (1 - fv) + (p01 * (1 - fu) + p11 * fu) * fv, n


def solid_from_grids(bm, outer, inner, mat_index=0):
    nv = len(outer) - 1
    nu = len(outer[0]) - 1
    vo = [[bm.verts.new(p) for p in row] for row in outer]
    vi = [[bm.verts.new(p) for p in row] for row in inner]
    for j in range(nv):
        for i in range(nu):
            bm.faces.new((vo[j][i], vo[j][i + 1], vo[j + 1][i + 1], vo[j + 1][i])).material_index = mat_index
            bm.faces.new((vi[j][i], vi[j + 1][i], vi[j + 1][i + 1], vi[j][i + 1])).material_index = mat_index
    border = [(0, i) for i in range(nu)] + [(j, nu) for j in range(nv)] + \
             [(nv, i) for i in range(nu, 0, -1)] + [(j, 0) for j in range(nv, 0, -1)]
    nxt = border[1:] + border[:1]
    for (j0, i0), (j1, i1) in zip(border, nxt):
        bm.faces.new((vo[j0][i0], vi[j0][i0], vi[j1][i1], vo[j1][i1])).material_index = mat_index


def pocket_cutter(name, grid, direction, depth, coll, mat, flat=None):
    """Closed solid that carves a recess of `depth` into the body behind a patch."""
    d = Vector(direction).normalized()
    outer = [[g[0] - d * 0.08 for g in row] for row in grid]
    if flat is None:
        inner = [[g[0] + d * depth for g in row] for row in grid]
    else:
        # flat floor: plane perpendicular to d at given distance along d
        inner = []
        for row in grid:
            r = []
            for g in row:
                p = g[0]
                t = flat - p.dot(d)
                r.append(p + d * t)
            inner.append(r)
    bm = bmesh.new()
    solid_from_grids(bm, outer, inner)
    return bm_to_object(bm, name, coll, [mat], smooth=False)


def lens_from_grid(name, grid, coll, mat, offset=0.0006, inset=0.0):
    nv = len(grid) - 1
    nu = len(grid[0]) - 1
    bm = bmesh.new()
    vs = [[bm.verts.new(g[0] + g[1] * offset) for g in row] for row in grid]
    for j in range(nv):
        for i in range(nu):
            bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
    ob = bm_to_object(bm, name, coll, [mat])
    return ob


# ---------------------------------------------------------------------------
# WHEELS
# ---------------------------------------------------------------------------


def tire_profile():
    pts = [(0.229, -0.100), (0.240, -0.112), (0.262, -0.1205), (0.285, -0.1215),
           (0.302, -0.115), (0.311, -0.102), (0.3148, -0.086)]
    centre = []
    for gy in (-0.055, -0.018):
        centre += [(0.315, gy - 0.007), (0.3075, gy - 0.005), (0.3075, gy + 0.005), (0.315, gy + 0.007)]
    mirror = [(r, -y) for (r, y) in reversed(centre)]
    tread = centre + mirror
    tail = [(r, -y) for (r, y) in reversed(pts)]
    return pts + tread + tail


def rim_barrel_profile():
    return [(0.247, 0.117), (0.246, 0.124), (0.238, 0.127), (0.213, 0.123), (0.208, 0.113),
            (0.206, 0.062), (0.200, -0.050), (0.204, -0.104), (0.238, -0.117), (0.241, -0.124),
            (0.232, -0.125), (0.225, -0.110), (0.222, 0.100), (0.232, 0.111)]


def build_spokes(name, coll, M, pairs=6):
    bm = bmesh.new()
    r0, r1 = 0.07, 0.214
    steps = 8
    for k in range(pairs):
        for side in (-1, 1):
            a = 2 * math.pi * k / pairs + side * math.radians(6.5)
            secs = []
            for s in range(steps + 1):
                t = s / steps
                r = lerp(r0, r1, t)
                # spokes converge at the hub
                aa = a - side * math.radians(6.5) * (1 - t) ** 1.6 * 0.85
                radial = Vector((math.cos(aa), 0, math.sin(aa)))
                tang = Vector((-math.sin(aa), 0, math.cos(aa)))
                w = lerp(0.026, 0.019, t)
                yf = lerp(0.024, 0.066, t ** 0.75)
                d = lerp(0.034, 0.024, t)
                c = radial * r
                q = [c - tang * w / 2 + Vector((0, yf, 0)),
                     c + tang * w / 2 + Vector((0, yf, 0)),
                     c + tang * w * 0.6 + Vector((0, yf - d, 0)),
                     c - tang * w * 0.6 + Vector((0, yf - d, 0))]
                secs.append([bm.verts.new(p) for p in q])
            for s in range(steps):
                A, B = secs[s], secs[s + 1]
                for e in range(4):
                    e1 = (e + 1) % 4
                    bm.faces.new((A[e], A[e1], B[e1], B[e]))
            bm.faces.new(list(reversed(secs[0])))
            bm.faces.new(secs[-1])
    ob = bm_to_object(bm, name, coll, [M["wheel"]], smooth=False)
    add_bevel(ob, 0.0018, 2, 30)
    return ob


def build_wheel(tag, coll, M, center, side, steer, camber, front):
    """side: +1 left (outer face +y), -1 right. Returns root empty."""
    root = bpy.data.objects.new(f"Wheel_{tag}", None)
    root.empty_display_type = "ARROWS"
    root.empty_display_size = 0.3
    coll.objects.link(root)
    root.location = center
    yaw = steer + (0 if side > 0 else math.pi)
    root.rotation_euler = (math.radians(camber), 0, yaw)

    parts = []
    tire = lathe(f"Tyre_{tag}", tire_profile(), coll, [M["tire"]], segs=128)
    add_subsurf(tire, 1, 2)
    parts.append(tire)
    barrel = lathe(f"RimBarrel_{tag}", rim_barrel_profile(), coll, [M["alu"], M["wheel"]], segs=96,
                   mat_fn=lambda j: 0 if j <= 3 else 1)
    add_subsurf(barrel, 1, 2)
    parts.append(barrel)
    parts.append(build_spokes(f"Spokes_{tag}", coll, M))
    hub = lathe(f"Hub_{tag}", [(0.022, 0.030), (0.080, 0.030), (0.088, 0.024), (0.088, -0.012), (0.022, -0.012)],
                coll, [M["wheel"]], segs=64)
    parts.append(hub)
    cap = lathe(f"CentreCap_{tag}", [(0.0005, 0.040), (0.018, 0.039), (0.026, 0.036), (0.028, 0.030),
                                     (0.028, 0.020), (0.0005, 0.020)], coll, [M["anodized"]], segs=48)
    add_subsurf(cap, 1, 2)
    parts.append(cap)
    # lug nuts (5 x 114.3)
    bm = bmesh.new()
    for k in range(5):
        a = 2 * math.pi * k / 5 + math.pi / 2
        p = Vector((0.0572 * math.cos(a), 0, 0.0572 * math.sin(a)))
        cylinder_bm(bm, p + Vector((0, 0.028, 0)), p + Vector((0, 0.052, 0)), 0.0105, 6)
        cylinder_bm(bm, p + Vector((0, 0.052, 0)), p + Vector((0, 0.058, 0)), 0.007, 12)
    nuts = bm_to_object(bm, f"LugNuts_{tag}", coll, [M["chrome"]], smooth=False)
    add_bevel(nuts, 0.0012, 1, 30)
    parts.append(nuts)
    # brake disc + bell
    disc = lathe(f"BrakeDisc_{tag}", [(0.100, -0.017), (0.176, -0.017), (0.178, -0.019), (0.178, -0.043),
                                      (0.176, -0.045), (0.100, -0.045)], coll, [M["disc"]], segs=96)
    parts.append(disc)
    bell = lathe(f"DiscBell_{tag}", [(0.060, 0.000), (0.102, -0.002), (0.104, -0.030), (0.096, -0.030),
                                     (0.094, -0.008), (0.060, -0.008)], coll, [M["anodized"]], segs=64)
    parts.append(bell)
    # caliper (does not spin) - sits at the trailing upper edge of the disc
    big = front
    prof = [(0.118, -0.068), (0.186, -0.068), (0.196, -0.058), (0.196, -0.002), (0.188, 0.008),
            (0.128, 0.008), (0.121, 0.000), (0.121, -0.013), (0.181, -0.013), (0.181, -0.049),
            (0.118, -0.049)]
    span = math.radians(62 if big else 46)
    mid = math.radians(125) if side > 0 else math.radians(55)
    cal = lathe(f"Caliper_{tag}", prof, coll, [M["caliper"]], segs=18, a0=mid - span / 2, a1=mid + span / 2)
    add_subsurf(cal, 1, 2)
    parts.append(cal)
    # upright / hub carrier hint
    upr = make_box(f"Upright_{tag}", (0, -0.075, 0), (0.12, 0.05, 0.22), coll, [M["anodized"]], bevel=0.008)
    parts.append(upr)
    for p in parts:
        p.parent = root
    return root, parts


# ---------------------------------------------------------------------------
# CAR
# ---------------------------------------------------------------------------


def side_y(bvh, x, z, default=0.9, sign=1):
    loc, _ = ray(bvh, (x, sign * 3.0, z), (0, -sign, 0), 4.0)
    return abs(loc.y) if loc is not None else default


def build_flare(name, bvh, coll, M, xa, sign):
    """Bolt-on over-fender following the wheel arch."""
    bm = bmesh.new()
    segs = 40
    th0, th1 = math.radians(-12), math.radians(192)
    prof_r = [ARCH_R + 0.085, ARCH_R + 0.06, ARCH_R + 0.022, ARCH_R - 0.004, ARCH_R - 0.006]
    prof_dy = [0.0, 0.032, 0.052, 0.048, -0.03]
    rings = []
    last = None
    for s in range(segs + 1):
        th = lerp(th0, th1, s / segs)
        ring = []
        for r, dy in zip(prof_r, prof_dy):
            x = xa + r * math.cos(th)
            z = ARCH_Z + r * math.sin(th)
            z = max(z, 0.12)
            ys = side_y(bvh, x, max(z, 0.2), last or 0.9, sign)
            if r == prof_r[0]:
                last = ys
            ring.append(bm.verts.new((x, sign * (ys + dy - 0.004), z)))
        rings.append(ring)
    n = len(prof_r)
    for s in range(segs):
        A, B = rings[s], rings[s + 1]
        for k in range(n):
            k1 = (k + 1) % n
            bm.faces.new((A[k], A[k1], B[k1], B[k]))
    bm.faces.new(rings[0])
    bm.faces.new(list(reversed(rings[-1])))
    ob = bm_to_object(bm, name, coll, [M["paint"]])
    add_subsurf(ob, 1, 2)
    # rivets
    bmr = bmesh.new()
    for k in range(9):
        th = lerp(th0 + 0.15, th1 - 0.15, k / 8)
        r = ARCH_R + 0.045
        x = xa + r * math.cos(th)
        z = ARCH_Z + r * math.sin(th)
        ys = side_y(bvh, x, max(z, 0.2), 0.9, sign) + 0.045
        m = Matrix.Translation((x, sign * ys, z)) @ Matrix.Rotation(math.radians(-90 * sign), 4, "X")
        bmesh.ops.create_uvsphere(bmr, u_segments=10, v_segments=6, radius=0.0065, matrix=m @ Matrix.Diagonal((1, 1, 0.45, 1)))
    rv = bm_to_object(bmr, name + "_Rivets", coll, [M["alu"]])
    return [ob, rv]


def build_side_skirt(name, bvh, coll, M, sign):
    bm = bmesh.new()
    x0 = FRONT_AXLE - ARCH_R - 0.03
    x1 = REAR_AXLE + ARCH_R + 0.03
    n = 24
    prof = [(-0.03, 0.245), (0.030, 0.230), (0.046, 0.150), (0.072, 0.118), (0.074, 0.105),
            (-0.05, 0.105)]
    rings = []
    for i in range(n + 1):
        x = lerp(x0, x1, i / n)
        ys = side_y(bvh, x, 0.24, 0.88, sign)
        rings.append([bm.verts.new((x, sign * (ys + dy), z)) for dy, z in prof])
    for i in range(n):
        A, B = rings[i], rings[i + 1]
        for k in range(len(prof)):
            k1 = (k + 1) % len(prof)
            bm.faces.new((A[k], A[k1], B[k1], B[k]))
    bm.faces.new(rings[0])
    bm.faces.new(list(reversed(rings[-1])))
    ob = bm_to_object(bm, name, coll, [M["carbon"]])
    add_subsurf(ob, 1, 2)
    return ob


def naca(code="6412", n=40):
    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100
    up, lo = [], []
    for i in range(n + 1):
        b = math.pi * i / n
        x = (1 - math.cos(b)) / 2
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        if x < p:
            yc = m / p ** 2 * (2 * p * x - x * x)
        else:
            yc = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x)
        up.append((x, yc + yt))
        lo.append((x, yc - yt))
    return list(reversed(up)) + lo[1:-1]


def build_wing(bvh, coll, M):
    objs = []
    chord, span = 0.30, 1.62
    xle, zw = -1.86, 1.235
    aoa = math.radians(9)
    foil = naca("6412", 36)
    bm = bmesh.new()
    secs = []
    for yy in (-span / 2, span / 2):
        ring = []
        for (fx, fz) in foil:
            # inverted foil (downforce), trailing edge raised by the angle of attack
            fz = -fz
            x = xle - fx * chord * math.cos(aoa) + fz * chord * math.sin(aoa)
            z = zw + fx * chord * math.sin(aoa) + fz * chord * math.cos(aoa)
            ring.append(bm.verts.new((x, yy, z)))
        secs.append(ring)
    n = len(foil)
    for k in range(n):
        k1 = (k + 1) % n
        bm.faces.new((secs[0][k], secs[0][k1], secs[1][k1], secs[1][k]))
    bm.faces.new(secs[0])
    bm.faces.new(list(reversed(secs[1])))
    wing = bm_to_object(bm, "Wing_Main", coll, [M["carbon"]])
    objs.append(wing)
    # gurney flap at the trailing edge
    te_x = xle - chord * math.cos(aoa)
    te_z = zw + chord * math.sin(aoa)
    objs.append(make_box("Wing_Gurney", (te_x + 0.004, 0, te_z + 0.012), (0.004, span - 0.01, 0.026),
                         coll, [M["anodized"]]))
    # end plates
    for s in (-1, 1):
        bm = bmesh.new()
        outline = [(-1.82, 1.29), (-1.85, 1.31), (-2.20, 1.33), (-2.23, 1.31), (-2.21, 1.17), (-2.10, 1.15),
                   (-1.87, 1.19)]
        vs0 = [bm.verts.new((x, s * (span / 2 + 0.001), z)) for x, z in outline]
        vs1 = [bm.verts.new((x, s * (span / 2 + 0.007), z)) for x, z in outline]
        bm.faces.new(vs0)
        bm.faces.new(list(reversed(vs1)))
        for k in range(len(outline)):
            k1 = (k + 1) % len(outline)
            bm.faces.new((vs0[k], vs0[k1], vs1[k1], vs1[k]))
        ep = bm_to_object(bm, f"Wing_EndPlate_{'L' if s > 0 else 'R'}", coll, [M["carbon"]], smooth=False)
        add_bevel(ep, 0.0025, 2, 30)
        objs.append(ep)
    # uprights from the boot lid
    for s in (-1, 1):
        y = s * 0.46
        loc, _ = ray(bvh, (-1.98, y, 3), (0, 0, -1))
        zb = loc.z if loc else 0.97
        bm = bmesh.new()
        outline = [(-1.93, zb - 0.01), (-2.07, zb - 0.01), (-2.02, zw + 0.03), (-1.92, zw + 0.03)]
        vs0 = [bm.verts.new((x, y - 0.005, z)) for x, z in outline]
        vs1 = [bm.verts.new((x, y + 0.005, z)) for x, z in outline]
        bm.faces.new(vs0)
        bm.faces.new(list(reversed(vs1)))
        for k in range(4):
            k1 = (k + 1) % 4
            bm.faces.new((vs0[k], vs0[k1], vs1[k1], vs1[k]))
        up = bm_to_object(bm, f"Wing_Upright_{'L' if s > 0 else 'R'}", coll, [M["anodized"]], smooth=False)
        add_bevel(up, 0.002, 2, 30)
        objs.append(up)
        objs.append(make_box(f"Wing_Base_{'L' if s > 0 else 'R'}", (-2.0, y, zb + 0.004), (0.17, 0.05, 0.012),
                             coll, [M["anodized"]], bevel=0.003))
    return objs


def build_interior(bvh, coll, M):
    objs = []
    drv = -0.37   # right-hand drive
    # dash
    dash = make_box("Dash", (0.70, 0, 0.70), (0.34, 1.30, 0.20), coll, [M["dash"]], bevel=0.03)
    objs.append(dash)
    objs.append(make_box("GaugeHood", (0.60, drv, 0.83), (0.16, 0.30, 0.07), coll, [M["suede"]],
                         rot=(0, math.radians(-10), 0), bevel=0.02))
    objs.append(make_box("Console", (0.25, 0.0, 0.42), (0.9, 0.22, 0.20), coll, [M["dash"]], bevel=0.02))
    objs.append(make_box("Floor", (-0.2, 0, 0.21), (2.2, 1.5, 0.02), coll, [M["dash"]]))
    # steering wheel (deep dish)
    sw_c = Vector((0.38, drv, 0.86))
    tilt = math.radians(112)
    rot = Matrix.Rotation(tilt, 4, "Y")
    rim = lathe("SteeringWheel_Rim", [(0.172 + 0.016 * math.cos(a), 0.016 * math.sin(a))
                                      for a in [i * 2 * math.pi / 12 for i in range(12)]],
                coll, [M["suede"]], segs=48)
    rim.matrix_world = Matrix.Translation(sw_c) @ rot @ Matrix.Rotation(math.radians(90), 4, "X")
    objs.append(rim)
    bm = bmesh.new()
    for a in (0, 120, 240):
        aa = math.radians(a + 90)
        p0 = Vector((0, 0, 0.05))
        p1 = Vector((0.165 * math.cos(aa), 0.165 * math.sin(aa), 0.0))
        cylinder_bm(bm, p0, p1, 0.011, 10)
    cylinder_bm(bm, (0, 0, 0.03), (0, 0, 0.075), 0.035, 24)
    sp = bm_to_object(bm, "SteeringWheel_Spokes", coll, [M["anodized"]])
    sp.matrix_world = Matrix.Translation(sw_c) @ rot
    objs.append(sp)
    bm = bmesh.new()
    cylinder_bm(bm, sw_c + rot.to_3x3() @ Vector((0, 0, 0.07)), Vector((0.70, drv, 0.72)), 0.028, 16)
    objs.append(bm_to_object(bm, "SteeringColumn", coll, [M["dash"]]))
    # bucket seats
    for y in (drv, -drv):
        objs += build_bucket_seat(f"Seat_{'R' if y < 0 else 'L'}", coll, M, Vector((-0.28, y, 0.25)))
    # hydraulic handbrake + shifter
    bm = bmesh.new()
    cylinder_bm(bm, (-0.02, -0.12, 0.52), (0.10, -0.12, 0.88), 0.012, 12)
    cylinder_bm(bm, (0.10, -0.12, 0.88), (0.115, -0.12, 0.93), 0.02, 16)
    cylinder_bm(bm, (0.15, 0.0, 0.52), (0.18, 0.0, 0.72), 0.008, 10)
    hb = bm_to_object(bm, "Handbrake_Shifter", coll, [M["alu_brushed"]])
    objs.append(hb)
    sk = bmesh.new()
    bmesh.ops.create_uvsphere(sk, u_segments=16, v_segments=10, radius=0.028,
                              matrix=Matrix.Translation((0.183, 0, 0.735)))
    objs.append(bm_to_object(sk, "ShiftKnob", coll, [M["anodized"]]))
    # roll cage
    objs += build_roll_cage(bvh, coll, M)
    return objs


def build_bucket_seat(name, coll, M, base):
    objs = []
    back_rot = (0, math.radians(-14), 0)
    objs.append(make_box(name + "_Cushion", base + Vector((0.12, 0, 0.10)), (0.48, 0.42, 0.10), coll,
                         [M["fabric"]], bevel=0.03))
    objs.append(make_box(name + "_Back", base + Vector((-0.15, 0, 0.45)), (0.10, 0.44, 0.74), coll,
                         [M["fabric"]], rot=back_rot, bevel=0.04))
    for s in (-1, 1):
        objs.append(make_box(name + f"_Bolster{s}", base + Vector((-0.10, s * 0.24, 0.40)), (0.24, 0.07, 0.62),
                             coll, [M["fabric_red"]], rot=back_rot, bevel=0.03))
        objs.append(make_box(name + f"_Wing{s}", base + Vector((-0.04, s * 0.18, 0.88)), (0.20, 0.06, 0.18),
                             coll, [M["fabric"]], rot=back_rot, bevel=0.025))
        objs.append(make_box(name + f"_Side{s}", base + Vector((0.12, s * 0.23, 0.17)), (0.44, 0.06, 0.16),
                             coll, [M["fabric_red"]], bevel=0.025))
        # harness belts
        objs.append(make_box(name + f"_Belt{s}", base + Vector((-0.05, s * 0.09, 0.62)), (0.012, 0.05, 0.55),
                             coll, [M["harness"]], rot=(0, math.radians(-30), 0)))
    for o in objs:
        add_subsurf(o, 1, 2)
    return objs


def build_roll_cage(bvh, coll, M):
    def inset(x, dy, dz, amount=0.075):
        o = Vector((x, 0, 0.75))
        d = Vector((0, dy, dz)).normalized()
        loc, _ = ray(bvh, o, d, 3)
        if loc is None:
            return o + d * 0.5
        return loc - d * amount

    mat = M["cage"]
    objs = []
    r = 0.019
    for s in (-1, 1):
        mh_top = inset(-0.62, s * 0.62, 0.55)
        fh_top = inset(0.15, s * 0.62, 0.55)
        a_mid = inset(0.55, s * 0.55, 0.40, 0.06)
        mh_bot = Vector((-0.62, s * 0.66, 0.22))
        fh_bot = Vector((0.82, s * 0.62, 0.32))
        objs.append(tube_curve(f"Cage_A_{s}", [fh_bot, a_mid, fh_top, mh_top], r, coll, mat))
        objs.append(tube_curve(f"Cage_Door_{s}", [mh_bot + Vector((0, 0, 0.35)), Vector((0.1, s * 0.71, 0.48)),
                                                  fh_bot + Vector((-0.05, 0, 0.20))], r, coll, mat))
        rear = Vector((-1.55, s * 0.58, 0.40))
        objs.append(tube_curve(f"Cage_RearStay_{s}", [mh_top, rear], r, coll, mat, smooth_corners=False))
    tl, tr = inset(-0.62, 0.62, 0.55), inset(-0.62, -0.62, 0.55)
    objs.append(tube_curve("Cage_MainHoop", [Vector((-0.62, 0.66, 0.22)), tl,
                                             (tl + tr) / 2 + Vector((0, 0, 0.02)), tr,
                                             Vector((-0.62, -0.66, 0.22))], r, coll, mat))
    objs.append(tube_curve("Cage_Diagonal", [tl, Vector((-0.62, -0.62, 0.30))], r, coll, mat, smooth_corners=False))
    fl, fr = inset(0.15, 0.62, 0.55), inset(0.15, -0.62, 0.55)
    objs.append(tube_curve("Cage_Header", [fl, (fl + fr) / 2 + Vector((0, 0, 0.02)), fr], r, coll, mat))
    return objs


def build_car(col_car, M):
    random.seed(CONFIG["seed"])
    body = build_body_shell(col_car, M)
    bvh = make_bvh(body)
    objs = []

    cut_coll = new_collection("Cutters")
    cutters = []

    # ---- wheel arches & wheel positions
    wheels_info = []
    for xa, front in ((FRONT_AXLE, True), (REAR_AXLE, False)):
        for sign in (1, -1):
            ys = side_y(bvh, xa, 0.50, 0.9, sign)
            wy = ys - 0.075
            wheels_info.append((xa, sign, wy, front))
            bm = bmesh.new()
            cylinder_bm(bm, (xa, sign * (wy - 0.27), ARCH_Z), (xa, sign * 1.6, ARCH_Z), ARCH_R, 72)
            cutters.append(bm_to_object(bm, f"Cut_Arch_{xa}_{sign}", cut_coll, [M["liner"]], smooth=False))

    # ---- panel gaps (thin slabs)
    gap = 0.0042
    slabs = []
    for s in (1, -1):
        slabs += [((0.865, s * 0.98, 0.50), (gap, 0.36, 0.66)),            # door front
                  ((-0.415, s * 0.98, 0.51), (gap, 0.36, 0.68)),           # door rear
                  ((1.52, s * 0.735, 0.85), (1.04, gap, 0.36)),           # hood sides
                  ((-1.85, s * 0.73, 1.0), (0.52, gap, 0.30))]            # boot sides
    slabs += [((0.995, 0, 0.90), (gap, 1.47, 0.22)),     # hood rear (cowl)
              ((2.04, 0, 0.82), (gap, 1.47, 0.36)),      # hood front
              ((-1.595, 0, 1.0), (gap, 1.46, 0.26)),     # boot front
              ((-2.25, 0, 0.885), (0.50, 1.46, gap))]    # boot rear lip
    bm = bmesh.new()
    for c, sz in slabs:
        box_bm(bm, c, sz)
    cutters.append(bm_to_object(bm, "Cut_PanelGaps", cut_coll, [M["gap"]], smooth=False))

    # ---- headlights
    lights_parts = []
    for s in (1, -1):
        A = (3.0, s * 0.43, 0.548)
        B = (3.0, s * 0.80, 0.560)
        Cc = (3.0, s * 0.80, 0.650)
        D = (3.0, s * 0.45, 0.622)
        corners = [A, B, Cc, D] if s > 0 else [B, A, D, Cc]
        g = patch_grid(bvh, corners, (-1, 0, 0), 16, 8)
        cutters.append(pocket_cutter(f"Cut_Head_{s}", g, (-1, 0, 0), 0.065, cut_coll, M["housing"]))
        lens = lens_from_grid(f"HeadLens_{s}", g, col_car, M["lens"])
        lights_parts.append(lens)
        # projectors
        for k, u in enumerate((0.28 if s > 0 else 0.72, 0.52 if s > 0 else 0.48)):
            p, n = grid_point(g, u, 0.55)
            c = p + Vector((-0.045, 0, 0))
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.026, matrix=Matrix.Translation(c))
            lights_parts.append(bm_to_object(bm, f"Projector_{s}_{k}", col_car, [M["projector"]]))
            ring = lathe(f"ProjectorRing_{s}_{k}", [(0.027, 0.0), (0.036, 0.0), (0.036, -0.03), (0.027, -0.03)],
                         col_car, [M["chrome"]], segs=32)
            ring.matrix_world = Matrix.Translation(c) @ Matrix.Rotation(math.radians(-90), 4, "Z")
            lights_parts.append(ring)
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.012,
                                      matrix=Matrix.Translation(c + Vector((-0.02, 0, 0))))
            lights_parts.append(bm_to_object(bm, f"ProjectorBulb_{s}_{k}", col_car, [M["led_white"]]))
        # LED daytime-running strip along the top edge (L-shaped signature)
        pts = []
        for i in range(13):
            u = i / 12
            p, n = grid_point(g, u, 0.86)
            pts.append(p + Vector((-0.012, 0, 0)))
        p0, _ = grid_point(g, 0.98 if s > 0 else 0.02, 0.86)
        p1, _ = grid_point(g, 0.98 if s > 0 else 0.02, 0.2)
        lights_parts.append(tube_curve(f"DRL_{s}", pts, 0.0045, col_car, M["led_white"], smooth_corners=True))
        lights_parts.append(tube_curve(f"DRL_Down_{s}", [p0 + Vector((-0.012, 0, 0)), p1 + Vector((-0.012, 0, 0))],
                                       0.0045, col_car, M["led_white"], smooth_corners=False))
        # amber side marker
        pa, _ = grid_point(g, 0.92 if s > 0 else 0.08, 0.35)
        bm = bmesh.new()
        box_bm(bm, pa + Vector((-0.02, 0, 0)), (0.01, 0.05, 0.012))
        lights_parts.append(bm_to_object(bm, f"Indicator_{s}", col_car, [M["led_amber"]], smooth=False))

    # ---- taillights + light bar
    for s in (1, -1):
        A = (-3.0, s * 0.42, 0.705)
        B = (-3.0, s * 0.80, 0.725)
        Cc = (-3.0, s * 0.80, 0.835)
        D = (-3.0, s * 0.42, 0.805)
        corners = [A, B, Cc, D] if s > 0 else [B, A, D, Cc]
        g = patch_grid(bvh, corners, (1, 0, 0), 16, 6)
        cutters.append(pocket_cutter(f"Cut_Tail_{s}", g, (1, 0, 0), 0.05, cut_coll, M["housing"]))
        lights_parts.append(lens_from_grid(f"TailLens_{s}", g, col_car, M["lens_red"]))
        for k, v in enumerate((0.25, 0.5, 0.75)):
            pts = []
            for i in range(11):
                u = 0.06 + 0.88 * i / 10
                p, n = grid_point(g, u, v)
                pts.append(p + Vector((0.025, 0, 0)))
            lights_parts.append(tube_curve(f"TailLED_{s}_{k}", pts, 0.0055, col_car, M["led_red"]))
    g = patch_grid(bvh, [(-3.0, 0.40, 0.745), (-3.0, -0.40, 0.745), (-3.0, -0.40, 0.79), (-3.0, 0.40, 0.79)],
                   (1, 0, 0), 16, 3)
    cutters.append(pocket_cutter("Cut_LightBar", g, (1, 0, 0), 0.03, cut_coll, M["housing"]))
    lights_parts.append(lens_from_grid("LightBarLens", g, col_car, M["lens_smoke"]))
    pts = [grid_point(g, i / 12, 0.5)[0] + Vector((0.015, 0, 0)) for i in range(13)]
    lights_parts.append(tube_curve("LightBarLED", pts, 0.003, col_car, M["led_red"]))

    # ---- front grille + intercooler + brake ducts
    g = patch_grid(bvh, [(3.0, 0.50, 0.235), (3.0, -0.50, 0.235), (3.0, -0.52, 0.44), (3.0, 0.52, 0.44)],
                   (-1, 0, 0), 20, 6)
    minx = min(c[0].x for row in g for c in row)
    cutters.append(pocket_cutter("Cut_Grille", g, (-1, 0, 0), 0.0, cut_coll, M["satin_black"], flat=-(minx - 0.11)))
    objs.append(build_honeycomb("Grille_Mesh", minx - 0.02, -0.53, 0.53, 0.225, 0.45, col_car, M["satin_black"]))
    ic = make_box("Intercooler", (minx - 0.07, 0, 0.335), (0.07, 1.04, 0.20), col_car, [M["intercooler"]])
    objs.append(ic)
    bm = bmesh.new()
    for i in range(70):
        y = -0.5 + i / 69
        box_bm(bm, (minx - 0.034, y, 0.335), (0.004, 0.0025, 0.19))
    objs.append(bm_to_object(bm, "Intercooler_Fins", col_car, [M["alu_brushed"]], smooth=False))
    for s in (1, -1):
        corners = [(3.0, s * 0.58, 0.24), (3.0, s * 0.78, 0.25), (3.0, s * 0.76, 0.36), (3.0, s * 0.59, 0.37)]
        if s < 0:
            corners = [corners[1], corners[0], corners[3], corners[2]]
        g = patch_grid(bvh, corners, (-1, 0, 0), 8, 5)
        mx = min(c[0].x for row in g for c in row)
        cutters.append(pocket_cutter(f"Cut_Duct_{s}", g, (-1, 0, 0), 0.0, cut_coll, M["satin_black"],
                                     flat=-(mx - 0.12)))
        objs.append(build_honeycomb(f"Duct_Mesh_{s}", mx - 0.02, s * 0.57, s * 0.79, 0.23, 0.38,
                                    col_car, M["satin_black"]))

    # ---- hood vent with louvres
    g = patch_grid(bvh, [(1.30, 0.30, 3), (1.30, -0.30, 3), (1.64, -0.26, 3), (1.64, 0.26, 3)], (0, 0, -1), 10, 8)
    cutters.append(pocket_cutter("Cut_HoodVent", g, (0, 0, -1), 0.06, cut_coll, M["satin_black"]))
    bm = bmesh.new()
    for i in range(9):
        v = 0.06 + 0.88 * i / 8
        p, n = grid_point(g, 0.5, v)
        rotm = Matrix.Rotation(math.radians(-38), 4, "Y")
        box_bm(bm, p + Vector((0, 0, -0.022)), (0.045, 0.56, 0.004), matrix=rotm)
    objs.append(bm_to_object(bm, "HoodVent_Louvres", col_car, [M["carbon"]], smooth=False))

    # ---- exhaust hole
    ex_y, ex_z = -0.52, 0.255
    loc, _ = ray(bvh, (-3.0, ex_y, ex_z), (1, 0, 0))
    ex_x = loc.x if loc else -2.27
    bm = bmesh.new()
    cylinder_bm(bm, (ex_x + 0.25, ex_y, ex_z), (ex_x - 0.3, ex_y, ex_z), 0.062, 48)
    cutters.append(bm_to_object(bm, "Cut_Exhaust", cut_coll, [M["soot"]], smooth=False))

    # ---- number plate recess
    g = patch_grid(bvh, [(-3.0, 0.27, 0.43), (-3.0, -0.27, 0.43), (-3.0, -0.27, 0.565), (-3.0, 0.27, 0.565)],
                   (1, 0, 0), 10, 4)
    cutters.append(pocket_cutter("Cut_Plate", g, (1, 0, 0), 0.018, cut_coll, M["satin_black"]))
    plate_g = g

    # ---- apply all cuts in one exact boolean
    mod = body.modifiers.new("Cuts", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    try:
        mod.solver = "EXACT"
    except TypeError:
        pass
    mod.operand_type = "COLLECTION"
    mod.collection = cut_coll
    try:
        mod.material_mode = "TRANSFER"
    except (AttributeError, TypeError):
        pass
    for c in cutters:
        c.hide_render = True
        c.display_type = "WIRE"
    bake_modifiers(body)
    for c in list(cut_coll.objects):
        bpy.data.objects.remove(c, do_unlink=True)
    bpy.data.collections.remove(cut_coll)
    objs.append(body)
    objs += lights_parts

    # ---- bolt-on parts
    for xa, front in ((FRONT_AXLE, True), (REAR_AXLE, False)):
        for s in (1, -1):
            objs += build_flare(f"OverFender_{'F' if front else 'R'}{'L' if s > 0 else 'R'}", bvh, col_car, M, xa, s)
    for s in (1, -1):
        objs.append(build_side_skirt(f"SideSkirt_{'L' if s > 0 else 'R'}", bvh, col_car, M, s))
    objs.append(build_splitter(bvh, col_car, M))
    objs += build_canards(bvh, col_car, M)
    objs += build_diffuser(bvh, col_car, M)
    objs += build_wing(bvh, col_car, M)
    objs += build_mirrors(bvh, col_car, M)
    objs += build_wipers(bvh, col_car, M)
    objs += build_exhaust(col_car, M, ex_x, ex_y, ex_z)
    objs += build_misc(bvh, col_car, M, plate_g)
    objs += build_interior(bvh, col_car, M)
    objs += build_livery(body, bvh, col_car, M)
    return body, objs, wheels_info


def build_honeycomb(name, x, y0, y1, z0, z1, coll, mat, cell=0.017, bar=0.0028):
    bm = bmesh.new()
    if y0 > y1:
        y0, y1 = y1, y0
    dy = math.sqrt(3) * cell
    dz = 1.5 * cell
    row = 0
    z = z0 - cell
    while z < z1 + cell:
        off = (dy / 2) if row % 2 else 0
        y = y0 - dy + off
        while y < y1 + dy:
            if not (y0 + cell * 0.5 <= y <= y1 - cell * 0.5 and z0 + cell * 0.8 <= z <= z1 - cell * 0.3):
                y += dy
                continue
            outer, inner = [], []
            for k in range(6):
                a = math.radians(60 * k + 30)
                outer.append(bm.verts.new((x, y + cell * math.cos(a), z + cell * math.sin(a))))
                inner.append(bm.verts.new((x, y + (cell - bar) * math.cos(a), z + (cell - bar) * math.sin(a))))
            for k in range(6):
                k1 = (k + 1) % 6
                bm.faces.new((outer[k], outer[k1], inner[k1], inner[k]))
            y += dy
        z += dz
        row += 1
    ob = bm_to_object(bm, name, coll, [mat], smooth=False)
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = 0.012
    return ob


def build_splitter(bvh, coll, M):
    bm = bmesh.new()
    pts = []
    n = 26
    for i in range(n + 1):
        y = lerp(-0.84, 0.84, i / n)
        loc, _ = ray(bvh, (3.0, y, 0.20), (-1, 0, 0))
        x = loc.x if loc else 2.0
        pts.append((x + 0.055, y))
    outline = list(reversed(pts)) + [(FRONT_AXLE + ARCH_R + 0.06, -0.84), (FRONT_AXLE + ARCH_R + 0.06, 0.84)]
    top = [bm.verts.new((x, y, 0.150)) for x, y in outline]
    bot = [bm.verts.new((x, y, 0.137)) for x, y in outline]
    bm.faces.new(top)
    bm.faces.new(list(reversed(bot)))
    for k in range(len(outline)):
        k1 = (k + 1) % len(outline)
        bm.faces.new((top[k], top[k1], bot[k1], bot[k]))
    ob = bm_to_object(bm, "Splitter", coll, [M["carbon"]], smooth=False)
    add_bevel(ob, 0.003, 2, 30)
    return ob


def build_canards(bvh, coll, M):
    objs = []
    for s in (1, -1):
        for k, z in enumerate((0.30, 0.385)):
            o = Vector((3.0, s * 1.6, z))
            loc, nor = ray(bvh, o, Vector((-1.0, -s * 0.75, 0)))
            if loc is None:
                continue
            nor = Vector((nor.x, nor.y, 0)).normalized()
            t = Vector((0, 0, 1)).cross(nor) * s
            if t.x > 0:
                t = -t
            length = 0.15 - 0.03 * k
            p0 = loc - nor * 0.01
            pts = [p0 - t * 0.01, p0 + t * length, p0 + nor * (0.075 - 0.015 * k) + t * 0.03]
            bm = bmesh.new()
            a = [bm.verts.new(p) for p in pts]
            b = [bm.verts.new(p + Vector((0, 0, -0.004))) for p in pts]
            bm.faces.new(a)
            bm.faces.new(list(reversed(b)))
            for i in range(3):
                i1 = (i + 1) % 3
                bm.faces.new((a[i], a[i1], b[i1], b[i]))
            ob = bm_to_object(bm, f"Canard_{s}_{k}", coll, [M["carbon"]], smooth=False)
            add_bevel(ob, 0.0015, 1, 30)
            objs.append(ob)
    return objs


def build_diffuser(bvh, coll, M):
    objs = []
    objs.append(make_box("Diffuser_Plate", (-2.10, 0, 0.175), (0.42, 1.30, 0.012), coll, [M["carbon"]],
                         rot=(0, math.radians(-9), 0)))
    bm = bmesh.new()
    for i in range(6):
        y = lerp(-0.55, 0.55, i / 5)
        vs = [(-1.92, 0.16), (-2.33, 0.21), (-2.33, 0.13), (-2.28, 0.115)]
        a = [bm.verts.new((x, y - 0.003, z)) for x, z in vs]
        b = [bm.verts.new((x, y + 0.003, z)) for x, z in vs]
        bm.faces.new(a)
        bm.faces.new(list(reversed(b)))
        for k in range(4):
            k1 = (k + 1) % 4
            bm.faces.new((a[k], a[k1], b[k1], b[k]))
    objs.append(bm_to_object(bm, "Diffuser_Fins", coll, [M["carbon"]], smooth=False))
    return objs


def build_mirrors(bvh, coll, M):
    objs = []
    for s in (1, -1):
        x = 0.70
        ys = side_y(bvh, x, 0.88, 0.82, s)
        base = Vector((x, s * (ys - 0.01), 0.875))
        tip = Vector((x - 0.03, s * (ys + 0.10), 0.95))
        bm = bmesh.new()
        cylinder_bm(bm, base, tip, 0.011, 16)
        objs.append(bm_to_object(bm, f"MirrorStalk_{s}", coll, [M["carbon"]]))
        bm = bmesh.new()
        m = Matrix.Translation(tip + Vector((-0.035, s * 0.04, 0.02))) @ Matrix.Diagonal((0.065, 0.10, 0.052, 1))
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0, matrix=m)
        # flatten the rear face where the glass sits
        for v in bm.verts:
            if v.co.x < tip.x - 0.035 - 0.02:
                v.co.x = tip.x - 0.035 - 0.02
        hous = bm_to_object(bm, f"MirrorHousing_{s}", coll, [M["carbon"]])
        objs.append(hous)
        bm = bmesh.new()
        box_bm(bm, (tip.x - 0.035 - 0.021, tip.y + s * 0.04, tip.z + 0.02), (0.002, 0.17, 0.08))
        mg = bm_to_object(bm, f"MirrorGlass_{s}", coll, [M["chrome"]], smooth=False)
        add_bevel(mg, 0.015, 4, 30)
        objs.append(mg)
    return objs


def build_wipers(bvh, coll, M):
    objs = []
    for k, (y0, y1) in enumerate(((-0.62, 0.02), (0.06, 0.66))):
        pts = []
        for i in range(9):
            y = lerp(y0, y1, i / 8)
            x = 0.885 - 0.03 * (i / 8)
            loc, nor = ray(bvh, (x, y, 3), (0, 0, -1))
            if loc is None:
                continue
            pts.append(loc + nor * 0.012)
        if len(pts) > 1:
            cu = tube_curve(f"Wiper_{k}", pts, 0.006, coll, M["satin_black"], smooth_corners=False)
            objs.append(cu)
    return objs


def build_exhaust(coll, M, x, y, z):
    prof = [(0.052, 0.08), (0.056, 0.08), (0.058, -0.04), (0.054, -0.055), (0.048, -0.05), (0.046, 0.08)]
    tip = lathe("Exhaust_Tip", prof, coll, [M["titanium"]], segs=48)
    tip.matrix_world = Matrix.Translation((x + 0.02, y, z)) @ Matrix.Rotation(math.radians(-90), 4, "Z")
    add_subsurf(tip, 1, 2)
    inner = lathe("Exhaust_Inner", [(0.046, 0.05), (0.046, 0.35)], coll, [M["soot"]], segs=32, closed=False)
    inner.matrix_world = tip.matrix_world.copy()
    return [tip, inner]


def build_misc(bvh, coll, M, plate_g):
    objs = []
    # front & rear tow straps
    for (x0, y, z, dirx, nm) in ((3.0, -0.60, 0.26, -1, "Front"), (-3.0, 0.62, 0.28, 1, "Rear")):
        loc, _ = ray(bvh, (x0, y, z), (dirx, 0, 0))
        if loc is None:
            continue
        bm = bmesh.new()
        m = Matrix.Translation(loc + Vector((-dirx * 0.03, 0, -0.03))) @ Matrix.Rotation(math.radians(90), 4, "Y") \
            @ Matrix.Rotation(math.radians(90), 4, "X") @ Matrix.Diagonal((1, 1.6, 1, 1))
        bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=0.035, radius2=0.035, depth=0.025, matrix=m)
        st = bm_to_object(bm, f"TowStrap_{nm}", coll, [M["strap"]])
        so = st.modifiers.new("Solidify", "SOLIDIFY")
        so.thickness = 0.004
        objs.append(st)
    # racing fuel cap on the rear quarter
    x, z = -1.02, 0.74
    ys = side_y(bvh, x, z, 0.9, -1)
    cap = lathe("FuelCap", [(0.0005, 0.010), (0.040, 0.009), (0.046, 0.004), (0.046, -0.004), (0.0005, -0.004)],
                coll, [M["alu"]], segs=48)
    cap.matrix_world = Matrix.Translation((x, -ys, z)) @ Matrix.Rotation(math.radians(180), 4, "Z")
    add_subsurf(cap, 1, 2)
    objs.append(cap)
    # rear number plate
    p, n = grid_point(plate_g, 0.5, 0.5)
    pl = make_box("NumberPlate", p + Vector((-0.012, 0, 0)), (0.004, 0.50, 0.115), coll, [M["plate"]], bevel=0.004)
    objs.append(pl)
    txt = text_object("PlateText", CONFIG["plate"], 0.075, coll, M["vinyl_black"])
    txt.matrix_world = basis_matrix((0, -1, 0), (-1, 0, 0), p + Vector((-0.015, 0, 0)))
    objs.append(txt)
    return objs


def text_object(name, body, size, coll, mat, extrude=0.0008):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.extrude = extrude
    cu.space_character = 1.05
    try:
        cu.font = bpy.data.fonts.load("<builtin>")
    except Exception:
        pass
    cu.offset_x = 0
    cu.materials.append(mat)
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    try:
        cu.body_format[0].use_bold = True
    except Exception:
        pass
    return ob


def shrinkwrap(ob, target, offset):
    m = ob.modifiers.new("Shrinkwrap", "SHRINKWRAP")
    m.target = target
    m.wrap_method = "NEAREST_SURFACEPOINT"
    m.wrap_mode = "ABOVE_SURFACE"
    m.offset = offset
    return m


def polar_disc(name, radius, coll, mat, rings=6, segs=48):
    bm = bmesh.new()
    c = bm.verts.new((0, 0, 0))
    prev = None
    for r in range(1, rings + 1):
        rr = radius * r / rings
        ring = [bm.verts.new((rr * math.cos(2 * math.pi * k / segs), rr * math.sin(2 * math.pi * k / segs), 0))
                for k in range(segs)]
        for k in range(segs):
            k1 = (k + 1) % segs
            if prev is None:
                bm.faces.new((c, ring[k], ring[k1]))
            else:
                bm.faces.new((prev[k], ring[k], ring[k1], prev[k1]))
        prev = ring
    return bm_to_object(bm, name, coll, [mat])


def build_livery(body, bvh, coll, M):
    objs = []
    # windshield sun strip
    g = patch_grid(bvh, [(0.24, 0.66, 3), (0.24, -0.66, 3), (0.40, -0.66, 3), (0.40, 0.66, 3)], (0, 0, -1), 20, 4)
    strip = lens_from_grid("SunStrip", g, coll, M["vinyl_black"], offset=0.0015)
    objs.append(strip)
    p, n = grid_point(g, 0.5, 0.42)
    t = text_object("SunStrip_Text", CONFIG["livery"], 0.105, coll, M["vinyl_white"])
    # the windshield faces forward, so readers in front see +y on their right
    t.matrix_world = basis_matrix((0, 1, 0), n, p + n * 0.0035)
    objs.append(t)
    # accent stripe under the text
    g2 = patch_grid(bvh, [(0.385, 0.66, 3), (0.385, -0.66, 3), (0.395, -0.66, 3), (0.395, 0.66, 3)], (0, 0, -1), 20, 1)
    objs.append(lens_from_grid("SunStrip_Accent", g2, coll, M["vinyl_accent"], offset=0.0022))
    # door roundels with race number
    for s in (1, -1):
        x, z = 0.22, 0.52
        loc, nor = ray(bvh, (x, s * 3, z), (0, -s, 0))
        if loc is None:
            continue
        disc = polar_disc(f"Roundel_{s}", 0.17, coll, M["vinyl_white"])
        disc.matrix_world = basis_matrix((-s, 0, 0), nor, loc + nor * 0.003)
        shrinkwrap(disc, body, 0.0016)
        objs.append(disc)
        num = text_object(f"Number_{s}", CONFIG["number"], 0.24, coll, M["vinyl_black"])
        num.matrix_world = basis_matrix((-s, 0, 0), nor, loc + nor * 0.006)
        shrinkwrap(num, body, 0.0032)
        objs.append(num)
        # quarter-panel lettering
        loc2, nor2 = ray(bvh, (-1.62, s * 3, 0.835), (0, -s, 0))
        if loc2 is not None:
            tt = text_object(f"Quarter_Text_{s}", "DRIFT SPEC", 0.045, coll, M["vinyl_white"])
            tt.matrix_world = basis_matrix((-s, 0, 0), nor2, loc2 + nor2 * 0.004)
            shrinkwrap(tt, body, 0.002)
            objs.append(tt)
        # side stripe along the skirt line
        g3 = patch_grid(bvh, [(1.0 if s < 0 else -0.85, s * 3, 0.29), (-0.85 if s < 0 else 1.0, s * 3, 0.29),
                              (-0.85 if s < 0 else 1.0, s * 3, 0.305), (1.0 if s < 0 else -0.85, s * 3, 0.305)],
                        (0, -s, 0), 30, 1)
        objs.append(lens_from_grid(f"SideStripe_{s}", g3, coll, M["vinyl_accent"], offset=0.0015))
    return objs


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------


def drift_geometry():
    """Turn centre & travel direction in the car's frame."""
    drift = math.radians(CONFIG["drift_deg"])
    d = Vector((math.cos(-drift), math.sin(-drift), 0))       # travel direction
    left = Vector((-d.y, d.x, 0))
    rear_mid = Vector((REAR_AXLE, 0, 0))
    centre = rear_mid + left * CONFIG["turn_radius"]
    return d, left, centre


def arc_strip(name, centre, radius, phi0, phi1, width, coll, mat, z=0.0015, n=160, attr=None, attr_fn=None):
    bm = bmesh.new()
    layer = bm.verts.layers.float.new(attr) if attr else None
    rows = []
    for i in range(n + 1):
        t = i / n
        phi = lerp(phi0, phi1, t)
        c, s = math.cos(phi), math.sin(phi)
        row = []
        for k in (-0.5, 0.5):
            r = radius + k * width
            v = bm.verts.new((centre.x + r * c, centre.y + r * s, z))
            if layer is not None:
                v[layer] = attr_fn(t, k)
            row.append(v)
        rows.append(row)
    for i in range(n):
        bm.faces.new((rows[i][0], rows[i + 1][0], rows[i + 1][1], rows[i][1]))
    ob = bm_to_object(bm, name, coll, [mat], smooth=True)
    return ob


def build_environment(coll, M, wheels_world, car_root):
    rng = random.Random(CONFIG["seed"])
    d, left, centre = drift_geometry()
    # ground
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=400)
    ground = bm_to_object(bm, "Ground_Asphalt", coll, [M["asphalt"]], smooth=False)

    # tyre marks: each wheel draws an arc around the turn centre, behind the car
    for (tag, pos, front) in wheels_world:
        rel = pos - centre
        r = rel.length
        phi = math.atan2(rel.y, rel.x)
        length = 1.15 if not front else 0.7
        strength = 1.0
        m = M["marks_light"] if front else M["marks"]
        arc_strip(f"TyreMark_{tag}", centre, r, phi - length, phi - 0.01, 0.24, coll, m,
                  z=0.0012 + rng.random() * 0.0004, attr="mark",
                  attr_fn=lambda t, k: clamp(t * 3.0, 0, 1) * (1.0 - abs(k) * 0.8) * strength)
    # older marks from previous runs
    for i in range(7):
        r = CONFIG["turn_radius"] + rng.uniform(-3.5, 3.5)
        phi = math.atan2(-left.y, -left.x) + rng.uniform(-0.6, 0.4)
        arc_strip(f"OldMark_{i}", centre, r, phi - rng.uniform(0.6, 1.4), phi + rng.uniform(0.2, 0.6), 0.22,
                  coll, M["marks_light"], z=0.0008, attr="mark", attr_fn=lambda t, k: math.sin(t * math.pi) * 0.9)

    # red/white curb on the inside of the corner
    phi_c = math.atan2(-left.y, -left.x)
    rc = CONFIG["turn_radius"] - 6.5
    bm = bmesh.new()
    layer = bm.verts.layers.float.new("curb_u")
    n = 220
    prof = [(-0.6, 0.0), (-0.55, 0.03), (0.0, 0.045), (0.55, 0.03), (0.6, 0.0)]
    rows = []
    for i in range(n + 1):
        phi = lerp(phi_c - 1.6, phi_c + 0.9, i / n)
        arc_len = rc * (phi - phi_c + 1.6)
        row = []
        for (w, h) in prof:
            r = rc + w
            v = bm.verts.new((centre.x + r * math.cos(phi), centre.y + r * math.sin(phi), h))
            v[layer] = arc_len / 1.0
            row.append(v)
        rows.append(row)
    for i in range(n):
        for k in range(len(prof) - 1):
            bm.faces.new((rows[i][k], rows[i + 1][k], rows[i + 1][k + 1], rows[i][k + 1]))
    bm_to_object(bm, "Curb", coll, [M["curb"]])
    # white track edge line outside the curb
    arc_strip("EdgeLine", centre, rc + 0.85, phi_c - 1.6, phi_c + 0.9, 0.12, coll, M["paint_line"], z=0.001, n=200)

    # concrete barriers on the outside of the corner
    rb = CONFIG["turn_radius"] + 9.0
    prof = [(-0.30, 0.0), (0.30, 0.0), (0.30, 0.08), (0.12, 0.30), (0.08, 0.81), (-0.08, 0.81),
            (-0.12, 0.30), (-0.30, 0.08)]
    seg_len = 3.8
    phi = phi_c - 2.2
    i = 0
    while phi < phi_c + 0.8:
        dphi = seg_len / rb
        bm = bmesh.new()
        secs = []
        for t in (0.0, 1.0):
            ph = phi + dphi * t * 0.985
            c, s = math.cos(ph), math.sin(ph)
            secs.append([bm.verts.new((centre.x + (rb + w) * c, centre.y + (rb + w) * s, h)) for w, h in prof])
        for k in range(len(prof)):
            k1 = (k + 1) % len(prof)
            bm.faces.new((secs[0][k], secs[0][k1], secs[1][k1], secs[1][k]))
        bm.faces.new(secs[0])
        bm.faces.new(list(reversed(secs[1])))
        ob = bm_to_object(bm, f"Barrier_{i}", coll, [M["concrete"]], smooth=False)
        add_bevel(ob, 0.01, 2, 30)
        phi += dphi
        i += 1
    # tyre stacks against the wall
    for j in range(9):
        ph = phi_c - 1.6 + j * 0.24 + rng.uniform(-0.03, 0.03)
        r = rb - 0.75
        base = Vector((centre.x + r * math.cos(ph), centre.y + r * math.sin(ph), 0))
        for h in range(rng.randint(3, 5)):
            t = lathe(f"TyreStack_{j}_{h}", [(0.20, -0.1), (0.31, -0.1), (0.32, 0.0), (0.31, 0.1), (0.20, 0.1),
                                             (0.19, 0.0)], coll, [M["tyre_wall"]], segs=32)
            t.matrix_world = Matrix.Translation(base + Vector((rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02),
                                                               0.1 + 0.2 * h))) @ Matrix.Rotation(math.radians(90), 4, "X")
    # catch fence + poles behind the wall
    rf = rb + 0.9
    bm = bmesh.new()
    ful = bm.verts.layers.float.new("fence_u")
    n = 160
    ph0, ph1 = phi_c - 2.2, phi_c + 0.8
    cols = []
    for i in range(n + 1):
        ph = lerp(ph0, ph1, i / n)
        col = [bm.verts.new((centre.x + rf * math.cos(ph), centre.y + rf * math.sin(ph), h)) for h in (0.0, 3.2)]
        for v in col:
            v[ful] = rf * (ph - ph0)
        cols.append(col)
    for i in range(n):
        bm.faces.new((cols[i][0], cols[i + 1][0], cols[i + 1][1], cols[i][1]))
    bm_to_object(bm, "CatchFence", coll, [M["chainlink"]], smooth=False)
    bm = bmesh.new()
    ph = ph0
    while ph < ph1:
        base = Vector((centre.x + rf * math.cos(ph), centre.y + rf * math.sin(ph), 0))
        cylinder_bm(bm, base, base + Vector((0, 0, 3.3)), 0.04, 12)
        ph += 3.0 / rf
    bm_to_object(bm, "FencePoles", coll, [M["galvanised"]])
    for h in (1.6, 3.2):
        pts = [Vector((centre.x + rf * math.cos(lerp(ph0, ph1, i / 60)), centre.y + rf * math.sin(lerp(ph0, ph1, i / 60)), h))
               for i in range(61)]
        tube_curve(f"FenceRail_{h}", pts, 0.022, coll, M["galvanised"], smooth_corners=False)
    # flood-light towers
    for j, ph in enumerate((phi_c - 1.3, phi_c - 0.25, phi_c + 0.55)):
        r = rf + 3.0
        base = Vector((centre.x + r * math.cos(ph), centre.y + r * math.sin(ph), 0))
        bm = bmesh.new()
        cylinder_bm(bm, base, base + Vector((0, 0, 16)), 0.16, 16)
        box_bm(bm, base + Vector((0, 0, 16.4)), (1.8, 1.8, 0.8))
        bm_to_object(bm, f"LightTower_{j}", coll, [M["galvanised"]])
    # distant tree line
    tex = bpy.data.textures.new("TreeClouds", "CLOUDS")
    tex.noise_scale = 0.9
    for i in range(120):
        ph = phi_c - 2.6 + 3.8 * i / 120 + rng.uniform(-0.02, 0.02)
        r = rf + rng.uniform(28, 70)
        base = Vector((centre.x + r * math.cos(ph), centre.y + r * math.sin(ph), 0))
        bm = bmesh.new()
        for b in range(3):
            rad = rng.uniform(1.6, 3.4)
            c = base + Vector((rng.uniform(-2, 2), rng.uniform(-2, 2), rad * 0.6 + rng.uniform(0.0, 3.0)))
            bmesh.ops.create_icosphere(bm, subdivisions=3, radius=rad,
                                       matrix=Matrix.Translation(c) @ Matrix.Diagonal((1, 1, rng.uniform(1.0, 1.6), 1)))
        tree = bm_to_object(bm, f"Tree_{i}", coll, [M["foliage"]])
        dm = tree.modifiers.new("Displace", "DISPLACE")
        dm.texture = tex
        dm.strength = rng.uniform(1.2, 2.0)
        dm.texture_coords = "GLOBAL"
    # tyre smoke volume trailing the rear wheels
    if CONFIG["smoke"]:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        smoke = bm_to_object(bm, "TyreSmoke", coll, [M["smoke"]], smooth=False)
        back = -d
        ang = math.atan2(back.y, back.x)
        # +x end of the volume sits at the rear tyres
        smoke.matrix_world = Matrix.Translation(Vector((REAR_AXLE, 0, 0.9)) + back * 4.6) @ \
            Matrix.Rotation(ang + math.pi, 4, "Z") @ Matrix.Diagonal((5.4, 2.9, 1.5, 1))
        smoke.visible_shadow = True
    return ground


def build_world(M):
    scn = bpy.context.scene
    world = bpy.data.worlds.new("Golden Hour")
    scn.world = world
    try:
        world.use_nodes = True
    except AttributeError:
        pass
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    for t in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING", "HOSEK_WILKIE"):
        try:
            sky.sky_type = t
            break
        except TypeError:
            continue
    sun_elev = math.radians(16)
    for attr, val in (("sun_disc", False), ("sun_elevation", sun_elev), ("sun_rotation", math.radians(150)),
                      ("altitude", 200.0), ("air_density", 1.2), ("dust_density", 2.5), ("aerosol_density", 2.5),
                      ("ozone_density", 1.0), ("turbidity", 3.0)):
        if hasattr(sky, attr):
            try:
                setattr(sky, attr, val)
            except Exception:
                pass
    L(nt, sky.outputs[0], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.2
    L(nt, bg.outputs[0], out.inputs["Surface"])

    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 5.0
    sun.angle = math.radians(0.8)
    sun.color = (1.0, 0.80, 0.60)
    so = bpy.data.objects.new("Sun", sun)
    scn.collection.objects.link(so)
    az = math.radians(62)
    so.rotation_euler = (0, math.radians(90) - sun_elev, az)
    return so


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def build_cameras(coll, car_root):
    cams = {}
    specs = {
        "hero": ((5.4, 3.55, 0.62), (0.15, 0.05, 0.50), 46, 2.8),
        "side": ((0.0, 7.0, 0.75), (-0.1, 0.0, 0.55), 55, 4.0),
        "rear": ((-5.6, -3.6, 0.85), (-0.5, 0.0, 0.6), 48, 2.8),
        "top": ((4.5, 4.5, 4.0), (-0.3, 0.0, 0.4), 40, 5.6),
        "wheel": ((2.35, 1.75, 0.42), (1.30, 0.80, 0.33), 60, 2.0),
    }
    for name, (loc, tgt, lens, fstop) in specs.items():
        cd = bpy.data.cameras.new(f"Cam_{name}")
        cd.lens = lens
        cd.sensor_width = 36
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.clip_start = 0.05
        cd.clip_end = 500
        co = bpy.data.objects.new(f"Cam_{name}", cd)
        coll.objects.link(co)
        co.location = car_root.matrix_world @ Vector(loc)
        look_at(co, car_root.matrix_world @ Vector(tgt))
        cd.dof.focus_distance = (Vector(loc) - Vector((0.6, 0.7, 0.5))).length if name == "hero" \
            else (Vector(loc) - Vector(tgt)).length
        cams[name] = co
    return cams


def setup_render(args):
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    cy = scn.cycles
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.01
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    cy.max_bounces = 12
    cy.glossy_bounces = 6
    cy.transmission_bounces = 12
    cy.transparent_max_bounces = 24
    cy.volume_bounces = 1
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 0.5
    try:
        cy.volume_step_rate = 2.0
    except AttributeError:
        pass
    if args.gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for t in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                try:
                    prefs.compute_device_type = t
                    prefs.get_devices()
                    if any(d.type == t for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = True
                        cy.device = "GPU"
                        break
                except TypeError:
                    continue
        except Exception:
            pass
    w, h = args.res
    scn.render.resolution_x = w
    scn.render.resolution_y = h
    scn.render.resolution_percentage = args.percent
    scn.render.film_transparent = False
    vs = scn.view_settings
    for vt in ("AgX", "Filmic"):
        try:
            vs.view_transform = vt
            break
        except TypeError:
            continue
    for look in ("AgX - Punchy", "Punchy", "AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            vs.look = look
            break
        except TypeError:
            continue
    vs.exposure = 0.0
    scn.render.image_settings.file_format = "PNG"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


class Args:
    render = None
    save = None
    samples = 384
    res = (1920, 1080)
    percent = 100
    smoke = True
    cam = "hero"
    gpu = False


def parse_args():
    a = Args()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    it = iter(argv)
    for tok in it:
        if tok == "--render":
            a.render = next(it)
        elif tok == "--save":
            a.save = next(it)
        elif tok == "--samples":
            a.samples = int(next(it))
        elif tok == "--res":
            w, h = next(it).lower().split("x")
            a.res = (int(w), int(h))
        elif tok == "--percent":
            a.percent = int(next(it))
        elif tok == "--no-smoke":
            a.smoke = False
        elif tok == "--cam":
            a.cam = next(it)
        elif tok == "--gpu":
            a.gpu = True
    return a


def clear_scene():
    """Remove everything this script would create (safe to run from the GUI)."""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.worlds, bpy.data.node_groups):
        for block in list(coll):
            coll.remove(block)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def main():
    args = parse_args()
    CONFIG["smoke"] = CONFIG["smoke"] and args.smoke
    clear_scene()
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"

    M = build_materials()
    col_car = new_collection("DriftCar")
    col_wheels = new_collection("Wheels", col_car)
    col_env = new_collection("Environment")
    col_cam = new_collection("Cameras")

    car_root = bpy.data.objects.new("CarRoot", None)
    car_root.empty_display_type = "PLAIN_AXES"
    col_car.objects.link(car_root)
    chassis = bpy.data.objects.new("Chassis", None)
    chassis.empty_display_type = "CUBE"
    chassis.empty_display_size = 0.2
    col_car.objects.link(chassis)
    chassis.parent = car_root

    body, parts, wheels_info = build_car(col_car, M)
    for o in parts:
        if o.parent is None:
            o.parent = chassis

    wheels_world = []
    for xa, sign, wy, front in wheels_info:
        tag = ("F" if front else "R") + ("L" if sign > 0 else "R")
        steer = math.radians(CONFIG["steer_deg"]) if front else 0.0
        camber = CONFIG["front_camber"] if front else CONFIG["rear_camber"]
        root, _ = build_wheel(tag, col_wheels, M, (xa, sign * wy, WHEEL_R), sign, steer, camber, front)
        root.parent = car_root
        wheels_world.append((tag, Vector((xa, sign * wy, 0)), front))

    # body roll (to the outside of the turn) and slight squat
    chassis.rotation_euler = (math.radians(CONFIG["roll_deg"]), math.radians(-0.4), 0)
    chassis.location = (0, 0, -0.004)

    build_environment(col_env, M, wheels_world, car_root)
    build_world(M)
    cams = build_cameras(col_cam, car_root)
    scn.camera = cams.get(args.cam, cams["hero"])
    setup_render(args)
    scn.frame_set(1)

    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
        print("saved", args.save)
    if args.render:
        scn.render.filepath = os.path.abspath(args.render)
        bpy.ops.render.render(write_still=True)
        print("rendered", args.render)


if __name__ == "__main__":
    main()
