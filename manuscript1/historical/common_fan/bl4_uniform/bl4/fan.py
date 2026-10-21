"""Deterministic midpoint refinements of the boundary of the 4D cross-polytope."""
from hashlib import sha256


def initial_fan():
    vertices = []
    for axis in range(4):
        for sign in (-1, 1):
            v = [0, 0, 0, 0]
            v[axis] = sign
            vertices.append(tuple(v))
    cells = [tuple(2 * axis + ((mask >> axis) & 1) for axis in range(4)) for mask in range(16)]
    return vertices, cells, 1


def refine_once(vertices, cells, scale):
    new_vertices = [tuple(2 * x for x in v) for v in vertices]
    edge_to_mid = {}

    def midpoint(i, j):
        edge = (i, j) if i < j else (j, i)
        if edge not in edge_to_mid:
            edge_to_mid[edge] = len(new_vertices)
            a, b = vertices[edge[0]], vertices[edge[1]]
            new_vertices.append(tuple(a[q] + b[q] for q in range(4)))
        return edge_to_mid[edge]

    out = []
    for a, b, c, d in cells:
        ab, ac, ad = midpoint(a, b), midpoint(a, c), midpoint(a, d)
        bc, bd, cd = midpoint(b, c), midpoint(b, d), midpoint(c, d)
        out.extend(((a, ab, ac, ad), (b, ab, bc, bd),
                    (c, ac, bc, cd), (d, ad, bd, cd),
                    (ab, cd, ac, ad), (ab, cd, ad, bd),
                    (ab, cd, bd, bc), (ab, cd, bc, ac)))
    return new_vertices, out, 2 * scale


def refine_selected(vertices, cells, scale, selected_indices):
    """Refine selected cells only, retaining a globally common denominator."""
    selected=set(map(int,selected_indices))
    if any(i<0 or i>=len(cells) for i in selected):
        raise ValueError("adaptive selection index out of range")
    new_vertices=[tuple(2*x for x in v) for v in vertices]
    coordinate_to_index={v:i for i,v in enumerate(new_vertices)}

    def midpoint(i,j):
        a,b=vertices[i],vertices[j]
        v=tuple(a[q]+b[q] for q in range(4))
        idx=coordinate_to_index.get(v)
        if idx is None:
            idx=len(new_vertices); coordinate_to_index[v]=idx; new_vertices.append(v)
        return idx

    out=[]
    for index,(a,b,c,d) in enumerate(cells):
        if index not in selected:
            out.append((a,b,c,d)); continue
        ab,ac,ad=midpoint(a,b),midpoint(a,c),midpoint(a,d)
        bc,bd,cd=midpoint(b,c),midpoint(b,d),midpoint(c,d)
        out.extend(((a,ab,ac,ad),(b,ab,bc,bd),(c,ac,bc,cd),(d,ad,bd,cd),
                    (ab,cd,ac,ad),(ab,cd,ad,bd),(ab,cd,bd,bc),(ab,cd,bc,ac)))
    return new_vertices,out,2*scale


def build_adaptive_fan(base_level,schedule):
    vertices,cells,scale=build_fan(base_level)
    for selected in schedule:
        vertices,cells,scale=refine_selected(vertices,cells,scale,selected)
    return vertices,cells,scale


def build_fan(level):
    if not isinstance(level, int) or level < 0:
        raise ValueError("level must be a nonnegative integer")
    vertices, cells, scale = initial_fan()
    for _ in range(level):
        vertices, cells, scale = refine_once(vertices, cells, scale)
    return vertices, cells, scale


def determinant4(columns):
    a = [[int(columns[c][r]) for c in range(4)] for r in range(4)]
    sign, previous = 1, 1
    for k in range(3):
        pivot = next((r for r in range(k, 4) if a[r][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            sign = -sign
        p = a[k][k]
        for i in range(k + 1, 4):
            for j in range(k + 1, 4):
                value = a[i][j] * p - a[i][k] * a[k][j]
                if k:
                    if value % previous:
                        raise AssertionError("non-exact Bareiss division")
                    value //= previous
                a[i][j] = value
        previous = p
    return sign * a[3][3]


def fan_digest(level, vertices, cells, scale):
    h = sha256()
    h.update(f"fixed_16cell_midpoint_v1,{level},{scale},{len(vertices)},{len(cells)}\n".encode())
    for v in vertices:
        h.update((",".join(map(str, v)) + "\n").encode())
    for c in cells:
        h.update((",".join(map(str, c)) + "\n").encode())
    return h.hexdigest()


def audit_fan(level, vertices, cells, scale):
    if len(cells) != 16 * 8**level or scale != 2**level:
        raise AssertionError("fan size/scale mismatch")
    if len(set(vertices)) != len(vertices):
        raise AssertionError("duplicate fan vertices")
    determinant_sum = 0
    for cell in cells:
        if len(set(cell)) != 4 or min(cell) < 0 or max(cell) >= len(vertices):
            raise AssertionError("invalid fan cell")
        det = determinant4([vertices[i] for i in cell])
        if not det:
            raise AssertionError("degenerate fan cell")
        determinant_sum += abs(det)
    if determinant_sum != 16 * scale**4:
        raise AssertionError("fan coverage/nonoverlap invariant failed")
    return {"level": level, "ray_count": len(vertices), "cell_count": len(cells),
            "direction_scale": scale, "direction_determinant_sum": determinant_sum,
            "fan_sha256": fan_digest(level, vertices, cells, scale)}


def adaptive_digest(base_level,schedule,vertices,cells,scale):
    h=sha256()
    h.update(f"adaptive_16cell_midpoint_v1,{base_level},{scale},{len(vertices)},{len(cells)}\n".encode())
    for round_selection in schedule:
        h.update((",".join(map(str,round_selection))+"\n").encode())
    for v in vertices: h.update((",".join(map(str,v))+"\n").encode())
    for c in cells: h.update((",".join(map(str,c))+"\n").encode())
    return h.hexdigest()


def audit_adaptive_fan(base_level,schedule,vertices,cells,scale):
    if scale != 2**(base_level+len(schedule)):
        raise AssertionError("adaptive scale mismatch")
    if len(set(vertices))!=len(vertices): raise AssertionError("duplicate adaptive vertices")
    determinant_sum=0
    for cell in cells:
        if len(set(cell))!=4 or min(cell)<0 or max(cell)>=len(vertices):
            raise AssertionError("invalid adaptive cell")
        det=determinant4([vertices[i] for i in cell])
        if not det: raise AssertionError("degenerate adaptive cell")
        determinant_sum+=abs(det)
    if determinant_sum!=16*scale**4:
        raise AssertionError("adaptive fan coverage/nonoverlap invariant failed")
    return {"fan_type":"adaptive","base_level":base_level,"round_count":len(schedule),
            "selection_counts":[len(x) for x in schedule],"ray_count":len(vertices),
            "cell_count":len(cells),"direction_scale":scale,
            "direction_determinant_sum":determinant_sum,
            "fan_sha256":adaptive_digest(base_level,schedule,vertices,cells,scale)}
