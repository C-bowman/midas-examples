from numpy import array, pi, sqrt
from tokamesh.construction import equilateral_mesh, Polygon, trim_vertices
from tokamesh import TriangularMesh
from tokamesh.tokamaks import mastu_boundary


# generate a mesh of equilateral triangles covering the full area
R_wall, z_wall = mastu_boundary()
R_centre = 0.5*(R_wall.min() + R_wall.max())
width = abs(z_wall).max() * 1.42
R, z, triangles = equilateral_mesh(
    R_range=(-width + R_centre, width + R_centre),
    z_range=(-width, width),
    rotation=pi / 6,
    resolution=0.06,
    pivot=(R_centre, 0.)
)

# now align a mesh vertex with the desired centre
R_min, R_max = 0.261, 1.55
z_min, z_max = -1.49, 1.49
z -= z[abs(z).argmin()]
R -= (R[abs(R - R_min).argmin()] - R_min)


# construct a filter to exclude vertices beyond the machine boundary
boundary_poly = Polygon(*mastu_boundary())
vertex_filter = (z > z_max) | (z < z_min) | (R > R_max) | (R + 1e-6 < R_min)

vertex_filter |= (
    ~boundary_poly.is_inside(R, z) &
    (boundary_poly.distance(R, z) > 1e-6)
)

# additionally use a hexagon mask to shape the top and bottom of the mesh
a = 0.5*sqrt(3)
unit_hexagon = [(0., 1.), (a, 0.5), (a, -0.5), (0., -1), (-a, -0.5), (-a, 0.5)]

hex_R, hex_z = [array([p[i] for p in unit_hexagon]) for i in [0, 1]]
hex_R *= z_max
hex_z *= z_max
hex_R += R[abs(R - 0.675).argmin()]

hex_poly = Polygon(hex_R, hex_z)
vertex_filter |= (
    ~hex_poly.is_inside(R, z) &
    (hex_poly.distance(R, z) > 1e-6)
)

# add a filter based on lines which follow the baffles
p1 = (0.893, 1.304)
p2 = (1.191, 1.007)
m = (p2[1] - p1[1]) / (p2[0] - p1[0])
c = p1[1] - m*p1[0]

vertex_filter |= z > R*m + c
vertex_filter |= z < -R*m - c

# apply the filter to get new mesh data
R, z, triangles = trim_vertices(R, z, triangles, trim_bools=vertex_filter)
mesh_data = (R, z, triangles)
mesh = TriangularMesh(R, z, triangles)






# if __name__ == "__main__":
#     from numpy import linspace
#     import matplotlib.pyplot as plt
#
#     # plot the mesh
#     plt.plot(*mastu_boundary(), ".-")
#     mesh.draw(plt)
#     plt.plot(hex_poly.x, hex_poly.y)
#
#     R_line = linspace(0., 2., 10)
#     plt.plot(R_line, R_line*m + c, c="red")
#     plt.plot(R_line, -R_line*m - c, c="red")
#
#     plt.axis("equal")
#     plt.show()