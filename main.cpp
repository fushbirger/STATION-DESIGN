#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

static float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    return len > 0 ? v * (1.0f / len) : Vec3();
}
static float radians(float deg) { return deg * 3.14159265358979f / 180.0f; }

// 4x4 column-major matrix
struct Mat4 {
    float m[16] = {0};
    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int c = 0; c < 4; c++)
            for (int row = 0; row < 4; row++) {
                float s = 0;
                for (int k = 0; k < 4; k++) s += m[k * 4 + row] * b.m[c * 4 + k];
                r.m[c * 4 + row] = s;
            }
        return r;
    }
};

static Mat4 translate(const Vec3& t) {
    Mat4 r = Mat4::identity();
    r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
    return r;
}
static Mat4 scale(const Vec3& s) {
    Mat4 r = Mat4::identity();
    r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
    return r;
}
static Mat4 rotateX(float rad) {
    Mat4 r = Mat4::identity();
    r.m[5] = std::cos(rad); r.m[6] = std::sin(rad);
    r.m[9] = -std::sin(rad); r.m[10] = std::cos(rad);
    return r;
}
static Mat4 rotateY(float rad) {
    Mat4 r = Mat4::identity();
    r.m[0] = std::cos(rad); r.m[8] = std::sin(rad);
    r.m[2] = -std::sin(rad); r.m[10] = std::cos(rad);
    return r;
}
static Mat4 rotateZ(float rad) {
    Mat4 r = Mat4::identity();
    r.m[0] = std::cos(rad); r.m[1] = std::sin(rad);
    r.m[4] = -std::sin(rad); r.m[5] = std::cos(rad);
    return r;
}
// Rotate a direction by the 3x3 part of m, or by its inverse (transpose) for pure rotations
static Vec3 mulDir(const Mat4& m, const Vec3& v) {
    return {m.m[0] * v.x + m.m[4] * v.y + m.m[8] * v.z, m.m[1] * v.x + m.m[5] * v.y + m.m[9] * v.z,
            m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z};
}
static Vec3 mulDirT(const Mat4& m, const Vec3& v) {
    return {m.m[0] * v.x + m.m[1] * v.y + m.m[2] * v.z, m.m[4] * v.x + m.m[5] * v.y + m.m[6] * v.z,
            m.m[8] * v.x + m.m[9] * v.y + m.m[10] * v.z};
}
static Mat4 perspective(float fovRad, float aspect, float zn, float zf) {
    Mat4 r;
    float f = 1.0f / std::tan(fovRad / 2.0f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zf + zn) / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zf * zn) / (zn - zf);
    return r;
}
static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

struct Camera {
    Vec3 position{0.0f, 2.0f, 8.0f};
    float yaw = -90.0f; // -90 = facing -Z
    float pitch = -10.0f;
    float fov = 60.0f;
    float speed = 5.0f;
    float sensitivity = 0.1f;

    Vec3 front() const {
        float cy = std::cos(radians(yaw)), sy = std::sin(radians(yaw));
        float cp = std::cos(radians(pitch)), sp = std::sin(radians(pitch));
        return normalize(Vec3(cy * cp, sp, sy * cp));
    }
    Vec3 right() const { return normalize(cross(front(), Vec3(0, 1, 0))); }

    Mat4 view() const { return lookAt(position, position + front(), Vec3(0, 1, 0)); }
    Mat4 projection(float aspect) const { return perspective(radians(fov), aspect, 0.1f, 200.0f); }

    void look(float dx, float dy) {
        yaw += dx * sensitivity;
        pitch -= dy * sensitivity;
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
    }
};

struct Mesh {
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei indexCount = 0;

    // Vertex format: [position(3), normal(3)]
    void upload(const std::vector<float>& vertices, const std::vector<unsigned>& indices) {
        indexCount = (GLsizei)indices.size();
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
    }
    void draw() const {
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    void destroy() {
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
        glDeleteVertexArrays(1, &vao);
    }
};

[[maybe_unused]] static Mesh makeSphere(int stacks = 32, int sectors = 64) {
    std::vector<float> v;
    std::vector<unsigned> idx;
    const float PI = 3.14159265358979f;
    for (int i = 0; i <= stacks; i++) {
        float phi = PI * i / stacks;
        for (int j = 0; j <= sectors; j++) {
            float theta = 2.0f * PI * j / sectors;
            float x = std::sin(phi) * std::cos(theta);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(theta);
            v.insert(v.end(), {x, y, z, x, y, z}); // normal = position
        }
    }
    for (int i = 0; i < stacks; i++) {
        for (int j = 0; j < sectors; j++) {
            unsigned a = i * (sectors + 1) + j;
            unsigned b = a + sectors + 1;
            idx.insert(idx.end(), {a, a + 1, b, a + 1, b + 1, b});
        }
    }
    Mesh m;
    m.upload(v, idx);
    return m;
}

static Mesh makePlane(float half = 50.0f) {
    std::vector<float> v = {
        -half, 0, -half, 0, 1, 0,
         half, 0, -half, 0, 1, 0,
         half, 0,  half, 0, 1, 0,
        -half, 0,  half, 0, 1, 0,
    };
    std::vector<unsigned> idx = {0, 3, 2, 0, 2, 1};
    Mesh m;
    m.upload(v, idx);
    return m;
}


// 1x1x1 cube centered at the origin ("square" shape)
static Mesh makeCube() {
    std::vector<float> v;
    std::vector<unsigned> idx;
    const float n[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int f = 0; f < 6; f++) {
        Vec3 N(n[f][0], n[f][1], n[f][2]);
        Vec3 up = std::fabs(N.y) > 0.5f ? Vec3(0, 0, 1) : Vec3(0, 1, 0);
        Vec3 a = normalize(cross(N, up)), b = cross(N, a);
        Vec3 c[4] = {N * 0.5f - a * 0.5f - b * 0.5f, N * 0.5f + a * 0.5f - b * 0.5f,
                     N * 0.5f + a * 0.5f + b * 0.5f, N * 0.5f - a * 0.5f + b * 0.5f};
        unsigned base = (unsigned)(v.size() / 6);
        for (auto& p : c) v.insert(v.end(), {p.x, p.y, p.z, N.x, N.y, N.z});
        idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    Mesh m;
    m.upload(v, idx);
    return m;
}

// Cylinder with diameter 1 and height 1 centered at the origin ("circle" shape)
static Mesh makeCylinder(int seg = 32) {
    std::vector<float> v;
    std::vector<unsigned> idx;
    const float PI = 3.14159265358979f;
    for (int i = 0; i <= seg; i++) { // side
        float t = 2.0f * PI * i / seg, x = std::cos(t), z = std::sin(t);
        v.insert(v.end(), {0.5f * x, -0.5f, 0.5f * z, x, 0, z, 0.5f * x, 0.5f, 0.5f * z, x, 0, z});
    }
    for (int i = 0; i < seg; i++) {
        unsigned a = i * 2;
        idx.insert(idx.end(), {a, a + 2, a + 1, a + 1, a + 2, a + 3});
    }
    for (int cap = 0; cap < 2; cap++) { // top & bottom caps
        float y = cap ? 0.5f : -0.5f, ny = cap ? 1.0f : -1.0f;
        unsigned c = (unsigned)(v.size() / 6);
        v.insert(v.end(), {0, y, 0, 0, ny, 0});
        for (int i = 0; i <= seg; i++) {
            float t = 2.0f * PI * i / seg;
            v.insert(v.end(), {0.5f * std::cos(t), y, 0.5f * std::sin(t), 0, ny, 0});
        }
        for (int i = 0; i < seg; i++) {
            if (cap) idx.insert(idx.end(), {c, c + 1 + i + 1, c + 1 + i});
            else     idx.insert(idx.end(), {c, c + 1 + i, c + 1 + i + 1});
        }
    }
    Mesh m;
    m.upload(v, idx);
    return m;
}

// Cone with base diameter 1 and height 1 centered at the origin, apex up (arrow heads)
static Mesh makeCone(int seg = 24) {
    std::vector<float> v;
    std::vector<unsigned> idx;
    const float PI = 3.14159265358979f;
    for (int i = 0; i <= seg; i++) {
        float t = 2.0f * PI * i / seg, x = std::cos(t), z = std::sin(t);
        Vec3 n = normalize(Vec3(x, 0.5f, z));
        v.insert(v.end(), {0.5f * x, -0.5f, 0.5f * z, n.x, n.y, n.z, 0, 0.5f, 0, n.x, n.y, n.z});
    }
    for (int i = 0; i < seg; i++) {
        unsigned a = i * 2;
        idx.insert(idx.end(), {a, a + 2, a + 1});
    }
    Mesh m;
    m.upload(v, idx);
    return m;
}

// Ring in the XZ plane (axis = Y), major radius 1, thin tube (rotation handles)
static Mesh makeTorus(int segs = 48, int sides = 10) {
    std::vector<float> v;
    std::vector<unsigned> idx;
    const float PI = 3.14159265358979f, minor = 0.03f;
    for (int i = 0; i <= segs; i++) {
        float th = 2.0f * PI * i / segs;
        for (int j = 0; j <= sides; j++) {
            float ph = 2.0f * PI * j / sides;
            Vec3 n(std::cos(ph) * std::cos(th), std::sin(ph), std::cos(ph) * std::sin(th));
            Vec3 p = Vec3(std::cos(th), 0, std::sin(th)) + n * minor;
            v.insert(v.end(), {p.x, p.y, p.z, n.x, n.y, n.z});
        }
    }
    for (int i = 0; i < segs; i++)
        for (int j = 0; j < sides; j++) {
            unsigned a = i * (sides + 1) + j, b = a + sides + 1;
            idx.insert(idx.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    Mesh m;
    m.upload(v, idx);
    return m;
}

static const char* VERT_SRC = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uProj * uView * world;
}
)";

static const char* FRAG_SRC = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uColor;
uniform vec3 uLightPos;
uniform vec3 uCamPos;
uniform bool uChecker;
uniform float uAlpha;

out vec4 FragColor;

void main() {
    vec3 base = uColor;
    if (uChecker) {
        float c = mod(floor(vWorldPos.x) + floor(vWorldPos.z), 2.0);
        base = mix(vec3(0.18), vec3(0.32), c);
    }

    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vWorldPos);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0) * (uChecker ? 0.0 : 0.6);

    vec3 color = base * (0.15 + diff) + vec3(spec);
    FragColor = vec4(pow(color, vec3(1.0 / 2.2)), uAlpha);
}
)";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader error: %s\n", log);
        std::exit(1);
    }
    return s;
}

static GLuint makeProgram() {
    GLuint vs = compile(GL_VERTEX_SHADER, VERT_SRC);
    GLuint fs = compile(GL_FRAGMENT_SHADER, FRAG_SRC);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Link error: %s\n", log);
        std::exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}


static Camera g_camera;
static bool g_firstMouse = true;
static double g_lastX = 0, g_lastY = 0;


// ---------- Scene objects ----------
struct Object { int shape; Vec3 pos, scale, color; Mat4 rot = Mat4::identity(); }; // shape = index into g_shapes
static std::vector<Object> g_objects;
static int g_selected = -1;                   // index into g_objects, -1 = none
static std::vector<Mesh> g_shapes;            // 0 = square, 1 = circle, 2+ = loaded models
static std::vector<std::string> g_shapeNames;
static const Vec3 PALETTE[] = {{0.85f, 0.85f, 0.85f}, {0.9f, 0.25f, 0.2f}, {0.2f, 0.6f, 0.95f},
                               {0.3f, 0.85f, 0.4f}, {0.95f, 0.8f, 0.2f}, {0.6f, 0.4f, 0.25f}};
static const int PALETTE_N = 6;
static int g_colorIdx = 0;
static Vec3 g_color = PALETTE[0];             // color used when placing

enum Mode { MODE_BUILD, MODE_REMOVE, MODE_MOVE, MODE_SIZE, MODE_ROTATE, MODE_FREECAM };
static Mode g_mode = MODE_BUILD, g_lastTool = MODE_BUILD;
static bool g_lmbPressed = false;             // left click this frame (not over the UI)
static bool g_rmbLook = false;                // right mouse held: look around

static const int GIZMO_AXES = 3;              // 3 = X, Y and Z handles (2 = only X and Y)
static const float GRID_SNAP = 0.25f;         // move / size step
static const float MIN_SCALE = 0.25f;
static const float ROT_SNAP = 3.14159265f / 12.0f; // 15 degrees

// Hotbar: one slot = a ready-made piece (shape + size + color)
struct Item { std::string name; int shape; Vec3 scale, color; };
static const char* CAT_NAMES[5] = {"WALLS", "FLOOR", "ROOF", "RAILS", "EXTRAS"};
static std::vector<Item> g_items[5];
static int g_cat = 0, g_item = 0;              // selected category / slot

// Stations: each one is a rectangular lot on the map with its own objects
struct Station { std::string name; float x0, z0, x1, z1; Vec3 color; std::vector<Object> objects; };
static std::vector<Station> g_stations;
static int g_cur = -1;                         // station being built, -1 = none
enum Scene { SCENE_MENU, SCENE_MAP, SCENE_BUILDER };
static Scene g_scene = SCENE_MENU;
static float g_mapX = 0, g_mapZ = 0, g_mapH = 60; // map camera (looks straight down)
static bool g_areaDrag = false;                // dragging out a new lot on the map
static int g_ax0 = 0, g_az0 = 0, g_ax1 = 0, g_az1 = 0;
static std::string g_msg;                      // message shown in the map window
static int g_hover = -1;                       // object under the cursor ("shine")

// First person camera: walk on the ground or fly freely
enum CamMove { CAM_WALK, CAM_FLY };
static CamMove g_camMove = CAM_WALK;
static float g_velY = 0;
static bool g_onGround = false;
static const float EYE_HEIGHT = 1.7f, GRAVITY = 20.0f, JUMP_SPEED = 7.0f;
static const float STEP_H = 0.5f, BODY_R = 0.3f, WALK_SPEED = 4.0f;

// Dragging out new stations is disabled for now: only 1 station
static const bool ALLOW_NEW_STATIONS = false;
static int g_mapSel = -1, g_hoverStation = -1;

struct Lot { const char* name; float x0, z0, x1, z1; };
static const Lot LOCKED_LOTS[] = {{"Station 2 (locked)", 22, -14, 38, -2},
                                  {"Station 3 (locked)", -38, 10, -22, 22}};
static bool lockedAt(int cx, int cz) {
    for (const Lot& l : LOCKED_LOTS)
        if (cx >= l.x0 && cx < l.x1 && cz >= l.z0 && cz < l.z1) return true;
    return false;
}


// ---------- Model loading (Wavefront .obj) ----------
#ifndef ASSET_DIR
#define ASSET_DIR "assets"
#endif

// Resolve a 1-based (or negative, relative) OBJ index to a 0-based index
static int objIndex(int i, size_t count) { return i > 0 ? i - 1 : (int)count + i; }

// Parses an .obj file into flat triangle data [position(3), normal(3)].
// Supports v / vn / f (v, v/vt, v//vn, v/vt/vn), n-gons and negative indices.
static bool loadOBJ(const std::string& path, std::vector<float>& verts, std::vector<unsigned>& idx) {
    std::ifstream in(path);
    if (!in) return false;
    std::vector<Vec3> pos, nrm;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            Vec3 p; ss >> p.x >> p.y >> p.z; pos.push_back(p);
        } else if (tag == "vn") {
            Vec3 n; ss >> n.x >> n.y >> n.z; nrm.push_back(n);
        } else if (tag == "f") {
            std::vector<int> pi, ni;
            std::string tok;
            while (ss >> tok) {
                int p = 0, t = 0, n = 0;
                if (std::sscanf(tok.c_str(), "%d/%d/%d", &p, &t, &n) == 3) {}
                else if (std::sscanf(tok.c_str(), "%d//%d", &p, &n) == 2) {}
                else if (std::sscanf(tok.c_str(), "%d/%d", &p, &t) == 2) {}
                else std::sscanf(tok.c_str(), "%d", &p);
                if (p == 0) continue;
                pi.push_back(objIndex(p, pos.size()));
                ni.push_back(n ? objIndex(n, nrm.size()) : -1);
            }
            for (size_t k = 1; k + 1 < pi.size(); k++) { // triangle fan
                size_t c[3] = {0, k, k + 1};
                bool ok = true;
                for (size_t q : c) if (pi[q] < 0 || pi[q] >= (int)pos.size()) ok = false;
                if (!ok) continue;
                Vec3 fn = normalize(cross(pos[pi[c[1]]] - pos[pi[c[0]]], pos[pi[c[2]]] - pos[pi[c[0]]]));
                for (size_t q : c) {
                    Vec3 p = pos[pi[q]];
                    Vec3 n = (ni[q] >= 0 && ni[q] < (int)nrm.size()) ? nrm[ni[q]] : fn;
                    idx.push_back((unsigned)(verts.size() / 6));
                    verts.insert(verts.end(), {p.x, p.y, p.z, n.x, n.y, n.z});
                }
            }
        }
    }
    if (verts.empty()) return false;

    // Fit the model into one grid cell: centered on X/Z, sitting on the cell floor
    const float INF = std::numeric_limits<float>::max();
    Vec3 mn(INF, INF, INF), mx(-INF, -INF, -INF);
    for (size_t i = 0; i < verts.size(); i += 6) {
        mn.x = std::min(mn.x, verts[i]);     mx.x = std::max(mx.x, verts[i]);
        mn.y = std::min(mn.y, verts[i + 1]); mx.y = std::max(mx.y, verts[i + 1]);
        mn.z = std::min(mn.z, verts[i + 2]); mx.z = std::max(mx.z, verts[i + 2]);
    }
    float ext = std::max(mx.x - mn.x, std::max(mx.y - mn.y, mx.z - mn.z));
    float k = ext > 0 ? 1.0f / ext : 1.0f;
    for (size_t i = 0; i < verts.size(); i += 6) {
        verts[i]     = (verts[i]     - (mn.x + mx.x) * 0.5f) * k;
        verts[i + 1] = (verts[i + 1] - mn.y) * k - 0.5f;
        verts[i + 2] = (verts[i + 2] - (mn.z + mx.z) * 0.5f) * k;
    }
    return true;
}

// Loads every .obj in <ASSET_DIR>/model and registers it as a placeable shape
static void loadModels() {
    namespace fs = std::filesystem;
    std::error_code ec;
    // Try the compiled-in path first, then common locations relative to the working directory
    const fs::path candidates[] = {fs::path(ASSET_DIR) / "model", fs::path("assets") / "model",
                                   fs::path("..") / "assets" / "model", fs::path("..") / ".." / "assets" / "model"};
    fs::path dir;
    bool found = false;
    for (const fs::path& c : candidates) {
        if (fs::is_directory(c, ec)) { dir = c; found = true; break; }
    }
    if (!found) {
        std::fprintf(stderr, "Model folder not found. Looked in:\n");
        for (const fs::path& c : candidates)
            std::fprintf(stderr, "  %s\n", fs::absolute(c, ec).string().c_str());
        return;
    }
    std::printf("Loading models from: %s\n", fs::absolute(dir, ec).string().c_str());
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (entry.is_regular_file() && ext == ".obj") files.push_back(entry.path());
    }
    if (files.empty()) std::fprintf(stderr, "No .obj files found in that folder.\n");
    std::sort(files.begin(), files.end());
    for (const fs::path& f : files) {
        std::vector<float> verts;
        std::vector<unsigned> idx;
        if (!loadOBJ(f.string(), verts, idx)) {
            std::fprintf(stderr, "Failed to load model: %s\n", f.string().c_str());
            continue;
        }
        Mesh m;
        m.upload(verts, idx);
        g_shapes.push_back(m);
        g_shapeNames.push_back(f.stem().string());
        std::printf("Loaded model: %s (%zu triangles)\n", f.filename().string().c_str(), idx.size() / 3);
    }
}

// ---------- Picking ----------
struct Ray { Vec3 o, d; };

// Ray from the camera through the mouse cursor
static Ray cursorRay(GLFWwindow* w) {
    double mx, my;
    int ww, wh;
    glfwGetCursorPos(w, &mx, &my);
    glfwGetWindowSize(w, &ww, &wh);
    if (wh == 0) wh = 1;
    float nx = 2.0f * (float)mx / ww - 1.0f, ny = 1.0f - 2.0f * (float)my / wh;
    float th = std::tan(radians(g_camera.fov) / 2.0f), aspect = (float)ww / wh;
    Vec3 f = g_camera.front(), r = g_camera.right(), u = cross(r, f);
    return {g_camera.position, normalize(f + r * (nx * th * aspect) + u * (ny * th))};
}

// Ray vs axis-aligned box. Returns entry distance and the face normal hit.
static bool rayAABB(const Ray& ray, const Vec3& mn, const Vec3& mx, float& tHit, Vec3& normal) {
    float tmin = -1e9f, tmax = 1e9f;
    Vec3 n;
    for (int i = 0; i < 3; i++) {
        if (std::fabs(ray.d[i]) < 1e-8f) {
            if (ray.o[i] < mn[i] || ray.o[i] > mx[i]) return false;
            continue;
        }
        float t1 = (mn[i] - ray.o[i]) / ray.d[i], t2 = (mx[i] - ray.o[i]) / ray.d[i], sign = -1.0f;
        if (t1 > t2) { std::swap(t1, t2); sign = 1.0f; }
        if (t1 > tmin) { tmin = t1; n = Vec3(); n[i] = sign; }
        tmax = std::min(tmax, t2);
        if (tmin > tmax) return false;
    }
    if (tmin < 0) return false; // camera inside the box
    tHit = tmin;
    normal = n;
    return true;
}

// Ray vs rotated box: the box is [mn, mx] in a frame centered at c and rotated by R
static bool rayBox(const Ray& ray, const Vec3& c, const Mat4& R, const Vec3& mn, const Vec3& mx, float& t, Vec3& n) {
    Ray local{mulDirT(R, ray.o - c), mulDirT(R, ray.d)};
    Vec3 nl;
    if (!rayAABB(local, mn, mx, t, nl)) return false;
    n = mulDir(R, nl);
    return true;
}

struct Hit { bool hit = false; float t = 0; Vec3 normal; int object = -1; }; // object -1 = ground

static Hit raycastScene(const Ray& ray) {
    Hit best;
    best.t = 200.0f;
    for (size_t i = 0; i < g_objects.size(); i++) {
        const Object& o = g_objects[i];
        Vec3 h = o.scale * 0.5f, n;
        float t;
        if (rayBox(ray, o.pos, o.rot, h * -1.0f, h, t, n) && t < best.t) best = {true, t, n, (int)i};
    }
    if (ray.d.y < -1e-6f) {
        float t = -ray.o.y / ray.d.y;
        if (t > 0 && t < best.t) best = {true, t, Vec3(0, 1, 0), -1};
    }
    return best;
}

// Where a new piece goes: on top of the face that was hit, or beside it, snapped to a 0.5 grid.
// False if the spot is outside the current station lot.
static bool placement(const Hit& h, const Ray& ray, const Item& it, Vec3& center) {
    if (!h.hit || g_cur < 0) return false;
    Vec3 p = ray.o + ray.d * h.t, half = it.scale * 0.5f, n = h.normal;
    auto sn = [](float v) { return std::round(v * 2.0f) / 2.0f; };
    if (n.y > 0.5f) {
        center = Vec3(sn(p.x), p.y + half.y, sn(p.z));
    } else if (n.y < -0.5f) {
        center = Vec3(sn(p.x), p.y - half.y, sn(p.z));
    } else { // side of an object: put the piece next to it, bottoms aligned
        const Object& o = g_objects[h.object];
        center = p + Vec3(n.x * half.x, 0, n.z * half.z);
        if (std::fabs(n.x) < 0.5f) center.x = sn(p.x);
        if (std::fabs(n.z) < 0.5f) center.z = sn(p.z);
        center.y = o.pos.y - o.scale.y * 0.5f + half.y;
    }
    const Station& st = g_stations[g_cur];
    return center.x >= st.x0 && center.x <= st.x1 && center.z >= st.z0 && center.z <= st.z1;
}

// Floor cell under the mouse (map mode)
static bool groundCell(const Ray& ray, int& cx, int& cz) {
    if (ray.d.y > -1e-6f) return false;
    Vec3 p = ray.o + ray.d * (-ray.o.y / ray.d.y);
    cx = (int)std::floor(p.x);
    cz = (int)std::floor(p.z);
    return cx >= -50 && cx < 50 && cz >= -50 && cz < 50;
}

// ---------- Gizmo (arrows for Move / Size, rings for Rotate) ----------
static const Vec3 AXIS_DIR[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
static const Vec3 AXIS_COLOR[3] = {{0.95f, 0.25f, 0.2f}, {0.3f, 0.85f, 0.4f}, {0.2f, 0.5f, 0.95f}};

static float snap(float v) { return std::round(v / GRID_SNAP) * GRID_SNAP; }
static float snapAngle(float a) { return std::round(a / ROT_SNAP) * ROT_SNAP; }
static Mat4 rotAboutAxis(int a, float ang) { return a == 0 ? rotateX(ang) : (a == 1 ? rotateY(ang) : rotateZ(ang)); }

// Position along the line (origin + t * D) that is closest to the mouse ray
static bool axisParam(const Ray& ray, const Vec3& origin, const Vec3& D, float& t) {
    Vec3 w0 = origin - ray.o;
    float b = dot(D, ray.d), d = dot(D, w0), e = dot(ray.d, w0), den = 1.0f - b * b;
    if (std::fabs(den) < 1e-4f) return false; // ray parallel to the axis
    t = (b * e - d) / den;
    return true;
}

// Where the mouse ray crosses the plane of a rotation ring (p is relative to the center c)
static bool ringPoint(const Ray& ray, const Vec3& c, int a, Vec3& p, float& t) {
    Vec3 n = AXIS_DIR[a];
    float den = dot(ray.d, n);
    if (std::fabs(den) < 1e-3f) return false;
    t = dot(c - ray.o, n) / den;
    if (t <= 0) return false;
    p = ray.o + ray.d * t - c;
    return true;
}
static float ringAngle(const Vec3& p, int a) {
    return std::atan2(dot(p, AXIS_DIR[(a + 2) % 3]), dot(p, AXIS_DIR[(a + 1) % 3]));
}

static float gizmoK() { // keeps handles a readable size at any camera distance
    Vec3 d = g_objects[g_selected].pos - g_camera.position;
    return std::max(0.5f, std::sqrt(dot(d, d)) * 0.1f);
}
// Size arrows follow the object's own (rotated) axes, Move arrows follow the world axes
static Mat4 gizmoFrame(const Object& o) { return g_mode == MODE_SIZE ? o.rot : Mat4::identity(); }
static float maxScale(const Object& o) { return std::max(o.scale.x, std::max(o.scale.y, o.scale.z)); }
static float gizmoLen(const Object& o, int axis, float k) {
    return (g_mode == MODE_SIZE ? o.scale[axis] : maxScale(o)) * 0.5f + k;
}
static float ringRadius(const Object& o, float k) { return 0.6f * maxScale(o) + 0.5f * k; }

// Which handle (axis 0..2) is under the mouse, or -1
static int pickGizmo(const Ray& ray) {
    if (g_selected < 0) return -1;
    const Object& o = g_objects[g_selected];
    float k = gizmoK(), best = 1e9f;
    int axis = -1;
    for (int a = 0; a < GIZMO_AXES; a++) {
        float t = 0;
        if (g_mode == MODE_ROTATE) {
            Vec3 p;
            if (ringPoint(ray, o.pos, a, p, t) && std::fabs(std::sqrt(dot(p, p)) - ringRadius(o, k)) < 0.15f * k && t < best) {
                best = t;
                axis = a;
            }
        } else {
            Vec3 pad = Vec3(0.12f, 0.12f, 0.12f) * k, mn = pad * -1.0f, mx = pad, n;
            mn[a] = 0;
            mx[a] = gizmoLen(o, a, k);
            if (rayBox(ray, o.pos, gizmoFrame(o), mn, mx, t, n) && t < best) { best = t; axis = a; }
        }
    }
    return axis;
}

struct Drag {
    bool active = false;
    int axis = 0;
    float t0 = 0, angle0 = 0;
    Vec3 pos, scale;
    Mat4 rot, frame;
};
static Drag g_drag;

static void beginDrag(const Ray& ray, int axis) {
    const Object& o = g_objects[g_selected];
    Drag d;
    d.active = true; d.axis = axis; d.pos = o.pos; d.scale = o.scale; d.rot = o.rot; d.frame = gizmoFrame(o);
    if (g_mode == MODE_ROTATE) {
        Vec3 p;
        float t;
        if (!ringPoint(ray, o.pos, axis, p, t)) return;
        d.angle0 = ringAngle(p, axis);
    } else if (!axisParam(ray, o.pos, mulDir(d.frame, AXIS_DIR[axis]), d.t0)) {
        return;
    }
    g_drag = d;
}

static void updateDrag(const Ray& ray) {
    Object& o = g_objects[g_selected];
    int a = g_drag.axis;
    if (g_mode == MODE_ROTATE) {
        Vec3 p;
        float t;
        if (!ringPoint(ray, g_drag.pos, a, p, t)) return;
        o.rot = rotAboutAxis(a, snapAngle(ringAngle(p, a) - g_drag.angle0)) * g_drag.rot;
        return;
    }
    Vec3 dir = mulDir(g_drag.frame, AXIS_DIR[a]);
    float t;
    if (!axisParam(ray, g_drag.pos, dir, t)) return;
    float delta = snap(t - g_drag.t0);
    if (g_mode == MODE_MOVE) {
        o.pos = g_drag.pos + dir * delta;
    } else { // Size: the pulled face follows the mouse, the opposite face stays put
        float sc = std::max(MIN_SCALE, g_drag.scale[a] + delta);
        o.scale[a] = sc;
        o.pos = g_drag.pos + dir * ((sc - g_drag.scale[a]) * 0.5f);
    }
}

// ---------- Ground walk helpers ----------
// Floor height under the feet: ground (0) or the top of an object that is low enough to step onto
static float floorHeightAt(const Vec3& feet) {
    float best = 0;
    for (const Object& o : g_objects) {
        Vec3 l = mulDirT(o.rot, feet - o.pos), h = o.scale * 0.5f;
        if (std::fabs(l.x) > h.x || std::fabs(l.z) > h.z) continue;
        float top = o.pos.y + h.y;
        if (top <= feet.y + STEP_H && top > best) best = top;
    }
    return best;
}
// Walls / tall objects that block the body
static bool blockedAt(const Vec3& feet) {
    for (const Object& o : g_objects) {
        Vec3 l = mulDirT(o.rot, feet - o.pos), h = o.scale * 0.5f;
        if (std::fabs(l.x) > h.x + BODY_R || std::fabs(l.z) > h.z + BODY_R) continue;
        float top = o.pos.y + h.y, bot = o.pos.y - h.y;
        if (top > feet.y + STEP_H && bot < feet.y + EYE_HEIGHT) return true;
    }
    return false;
}
static void setCamMove(CamMove m) {
    g_camMove = m;
    g_velY = 0;
    g_onGround = false;
}

// ---------- Mode / selection ----------
static void setMode(GLFWwindow* w, Mode m) {
    if (m != MODE_FREECAM) g_lastTool = m;
    g_mode = m;
    g_drag.active = false;
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // the cursor stays visible in every mode
    g_firstMouse = true;
}

static void deleteObject(int i) {
    if (i < 0 || i >= (int)g_objects.size()) return;
    g_objects.erase(g_objects.begin() + i);
    if (g_selected == i) g_selected = -1;
    else if (g_selected > i) g_selected--;
    g_hover = -1;
    g_drag.active = false;
}
static void deleteSelected() { deleteObject(g_selected); }

static void selectItem(GLFWwindow* w, int cat, int idx) {
    g_cat = cat;
    g_item = idx;
    g_color = g_items[cat][idx].color;
    setMode(w, MODE_BUILD); // picking a piece = taking the blueprint in hand
}

// Hotbar contents. Models loaded from assets/model go into EXTRAS.
static void buildItems() {
    auto add = [](int c, const char* n, int sh, Vec3 sc, Vec3 col) { g_items[c].push_back({n, sh, sc, col}); };
    add(0, "Wall",       0, Vec3(4, 3, 0.3f),      Vec3(0.85f, 0.85f, 0.85f));
    add(0, "Tall wall",  0, Vec3(4, 5, 0.3f),      Vec3(0.8f, 0.8f, 0.85f));
    add(0, "Pillar",     1, Vec3(0.6f, 3, 0.6f),   Vec3(0.8f, 0.75f, 0.7f));
    add(1, "Floor tile", 0, Vec3(2, 0.2f, 2),      Vec3(0.6f, 0.6f, 0.65f));
    add(1, "Platform",   0, Vec3(6, 0.4f, 3),      Vec3(0.7f, 0.7f, 0.72f));
    add(1, "Round floor",1, Vec3(3, 0.2f, 3),      Vec3(0.6f, 0.65f, 0.6f));
    add(2, "Flat roof",  0, Vec3(4, 0.25f, 4),     Vec3(0.55f, 0.3f, 0.3f));
    add(2, "Wide roof",  0, Vec3(8, 0.25f, 6),     Vec3(0.5f, 0.28f, 0.28f));
    add(2, "Round roof", 1, Vec3(4, 0.3f, 4),      Vec3(0.5f, 0.35f, 0.3f));
    add(3, "Track",      0, Vec3(1.5f, 0.12f, 4),  Vec3(0.35f, 0.3f, 0.28f));
    add(3, "Long track", 0, Vec3(1.5f, 0.12f, 8),  Vec3(0.35f, 0.3f, 0.28f));
    add(3, "Buffer stop",0, Vec3(1.2f, 0.8f, 0.4f),Vec3(0.9f, 0.25f, 0.2f));
    add(4, "Block",      0, Vec3(1, 1, 1),         Vec3(0.85f, 0.85f, 0.85f));
    add(4, "Cylinder",   1, Vec3(1, 1, 1),         Vec3(0.85f, 0.85f, 0.85f));
    for (int i = 2; i < (int)g_shapes.size(); i++)
        add(4, g_shapeNames[i].c_str(), i, Vec3(1, 1, 1), Vec3(0.85f, 0.85f, 0.85f));
    g_color = g_items[0][0].color;
}

// ---------- Map <-> builder ----------
static void enterStation(GLFWwindow* w, int i) {
    g_cur = i;
    g_objects = std::move(g_stations[i].objects);
    g_stations[i].objects.clear();
    g_selected = -1;
    g_hover = -1;
    g_msg.clear();
    g_scene = SCENE_BUILDER;
    setMode(w, MODE_BUILD);
    const Station& st = g_stations[i];
    float cx = (st.x0 + st.x1) * 0.5f, cz = (st.z0 + st.z1) * 0.5f, e = std::max(st.x1 - st.x0, st.z1 - st.z0);
    g_velY = 0; g_onGround = false;
    if (g_camMove == CAM_WALK) {
        g_camera.position = Vec3(cx, EYE_HEIGHT, st.z1 - 1.0f);
        g_camera.pitch = -10.0f;
    } else {
        g_camera.position = Vec3(cx, 4.0f + e * 0.5f, cz + 6.0f + e * 0.8f);
        g_camera.pitch = -30.0f;
    }
    g_camera.yaw = -90.0f;
}

static void exitToMap(GLFWwindow* w) {
    if (g_cur >= 0) {
        Station& st = g_stations[g_cur];
        st.objects = std::move(g_objects);
        g_mapX = (st.x0 + st.x1) * 0.5f;
        g_mapZ = (st.z0 + st.z1) * 0.5f;
        g_mapSel = g_cur;
    }
    g_objects.clear();
    g_cur = -1;
    g_selected = -1;
    g_hover = -1;
    g_mapH = 40.0f;
    g_scene = SCENE_MAP;
    setMode(w, MODE_BUILD);
}

// Called when the mouse is released after dragging out a lot on the map
static void finishArea(GLFWwindow* w) {
    int x0 = std::min(g_ax0, g_ax1), x1 = std::max(g_ax0, g_ax1) + 1;
    int z0 = std::min(g_az0, g_az1), z1 = std::max(g_az0, g_az1) + 1;
    if (x1 - x0 < 2 || z1 - z0 < 2) { g_msg = "Area is too small (minimum 2x2 cells)."; return; }
    for (const Station& st : g_stations)
        if (x0 < st.x1 && x1 > st.x0 && z0 < st.z1 && z1 > st.z0) { g_msg = "That area overlaps another station."; return; }
    Station st;
    st.name = "Station " + std::to_string(g_stations.size() + 1);
    st.x0 = (float)x0; st.x1 = (float)x1; st.z0 = (float)z0; st.z1 = (float)z1;
    st.color = PALETTE[g_stations.size() % PALETTE_N];
    g_stations.push_back(st);
    enterStation(w, (int)g_stations.size() - 1); // a new area opens the builder
}

// ---------- Input callbacks ----------
static void mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
    bool overUI = ImGui::GetIO().WantCaptureMouse;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !overUI) g_lmbPressed = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS && g_scene == SCENE_BUILDER && g_mode != MODE_FREECAM && !overUI) { // hold RMB to turn the camera
            g_rmbLook = true;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            g_firstMouse = true;
        } else if (action == GLFW_RELEASE && g_rmbLook) {
            g_rmbLook = false;
            if (g_mode != MODE_FREECAM) glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS || ImGui::GetIO().WantCaptureKeyboard) return;
    if (key == GLFW_KEY_ESCAPE) { // builder -> map -> menu -> quit
        if (g_scene == SCENE_BUILDER) exitToMap(w);
        else if (g_scene == SCENE_MAP) g_scene = SCENE_MENU;
        else glfwSetWindowShouldClose(w, 1);
        return;
    }
    if (g_scene != SCENE_BUILDER) return;
    int n = (int)g_items[g_cat].size();
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_5) selectItem(w, key - GLFW_KEY_1, 0); // category
    if (key == GLFW_KEY_E) selectItem(w, g_cat, (g_item + 1) % n);                  // next piece
    if (key == GLFW_KEY_Q) selectItem(w, g_cat, (g_item + n - 1) % n);             // previous piece
    if (key == GLFW_KEY_C) { g_colorIdx = (g_colorIdx + 1) % PALETTE_N; g_color = PALETTE[g_colorIdx]; }
    if (key == GLFW_KEY_F) setCamMove(g_camMove == CAM_WALK ? CAM_FLY : CAM_WALK); // Ground walk <-> Free cam
    if (key == GLFW_KEY_TAB) setMode(w, g_mode == MODE_FREECAM ? g_lastTool : MODE_FREECAM);
    if (key == GLFW_KEY_DELETE) deleteSelected();
}

static void mouseCallback(GLFWwindow*, double x, double y) {
    if (g_firstMouse) { g_lastX = x; g_lastY = y; g_firstMouse = false; }
    bool freeLook = g_mode == MODE_FREECAM && !ImGui::GetIO().WantCaptureMouse; // not while over the UI
    if (g_scene == SCENE_BUILDER && (freeLook || g_rmbLook)) g_camera.look((float)(x - g_lastX), (float)(y - g_lastY));
    g_lastX = x;
    g_lastY = y;
}

static void scrollCallback(GLFWwindow*, double, double yoff) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (g_scene == SCENE_MENU) return;
    if (g_scene == SCENE_MAP) { // zoom the map
        g_mapH = std::min(120.0f, std::max(8.0f, g_mapH * (1.0f - (float)yoff * 0.1f)));
        return;
    }
    g_camera.fov -= (float)yoff * 2.0f;
    if (g_camera.fov < 20.0f) g_camera.fov = 20.0f;
    if (g_camera.fov > 100.0f) g_camera.fov = 100.0f;
}

static void processInput(GLFWwindow* w, float dt) {
    if (ImGui::GetIO().WantTextInput) return; // typing into a UI field
    if (g_scene == SCENE_MENU) return;

    float v = g_camera.speed * dt;
    if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) v *= 3.0f;

    if (g_scene == SCENE_MAP) { // pan the map
        v *= g_mapH * 0.15f;
        if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) g_mapZ -= v;
        if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) g_mapZ += v;
        if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) g_mapX -= v;
        if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) g_mapX += v;
        g_mapX = std::min(60.0f, std::max(-60.0f, g_mapX));
        g_mapZ = std::min(60.0f, std::max(-60.0f, g_mapZ));
        return;
    }

    auto down = [&](int k) { return glfwGetKey(w, k) == GLFW_PRESS; };
    if (g_camMove == CAM_FLY) { // Free cam
        Vec3 f = g_camera.front(), r = g_camera.right();
        if (down(GLFW_KEY_W)) g_camera.position += f * v;
        if (down(GLFW_KEY_S)) g_camera.position -= f * v;
        if (down(GLFW_KEY_D)) g_camera.position += r * v;
        if (down(GLFW_KEY_A)) g_camera.position -= r * v;
        if (down(GLFW_KEY_SPACE)) g_camera.position += Vec3(0, 1, 0) * v;
        if (down(GLFW_KEY_LEFT_CONTROL)) g_camera.position -= Vec3(0, 1, 0) * v;
        return;
    }

    // Ground walk: walk on the ground, gravity, jump, small steps, collision with objects
    Vec3 f = g_camera.front(); f.y = 0; f = normalize(f);
    Vec3 r = g_camera.right(); r.y = 0; r = normalize(r);
    Vec3 mv;
    if (down(GLFW_KEY_W)) mv += f;
    if (down(GLFW_KEY_S)) mv -= f;
    if (down(GLFW_KEY_D)) mv += r;
    if (down(GLFW_KEY_A)) mv -= r;
    if (dot(mv, mv) > 0) mv = normalize(mv) * (WALK_SPEED * (down(GLFW_KEY_LEFT_SHIFT) ? 1.8f : 1.0f) * dt);

    Vec3 feet = g_camera.position - Vec3(0, EYE_HEIGHT, 0);
    Vec3 tx = feet + Vec3(mv.x, 0, 0);
    if (!blockedAt(tx) || blockedAt(feet)) feet = tx;
    Vec3 tz = feet + Vec3(0, 0, mv.z);
    if (!blockedAt(tz) || blockedAt(feet)) feet = tz;

    if (g_onGround && down(GLFW_KEY_SPACE)) { g_velY = JUMP_SPEED; g_onGround = false; }
    g_velY -= GRAVITY * dt;
    feet.y += g_velY * dt;
    float fl = floorHeightAt(feet);
    if (feet.y <= fl) { feet.y = fl; g_velY = 0; g_onGround = true; }
    else g_onGround = false;

    g_camera.position = feet + Vec3(0, EYE_HEIGHT, 0);
}

// World point -> screen pixels (for labels)
static bool projectToScreen(const Vec3& p, const Mat4& view, const Mat4& proj, ImVec2 screen, ImVec2& out) {
    Mat4 vp = proj * view;
    float x = vp.m[0] * p.x + vp.m[4] * p.y + vp.m[8] * p.z + vp.m[12];
    float y = vp.m[1] * p.x + vp.m[5] * p.y + vp.m[9] * p.z + vp.m[13];
    float cw = vp.m[3] * p.x + vp.m[7] * p.y + vp.m[11] * p.z + vp.m[15];
    if (cw <= 0.001f) return false;
    out = ImVec2((x / cw * 0.5f + 0.5f) * screen.x, (1.0f - (y / cw * 0.5f + 0.5f)) * screen.y);
    return true;
}

// ---------- UI ----------
static void drawMenuUI(GLFWwindow* w) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(380, 0));
    ImGui::Begin("Main Menu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    ImGui::SetWindowFontScale(2.2f);
    ImVec2 ts = ImGui::CalcTextSize("Station Design");
    ImGui::SetCursorPosX((380 - ts.x) * 0.5f);
    ImGui::TextUnformatted("Station Design");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Spacing();
    if (ImGui::Button("Start", ImVec2(-1, 44))) {
        g_scene = SCENE_MAP;
        g_mapX = g_mapZ = 0; g_mapH = 40.0f; g_mapSel = -1; g_msg.clear();
        setMode(w, MODE_BUILD);
    }
    if (ImGui::CollapsingHeader("Controls")) {
        ImGui::TextWrapped("Map: click a station, then Build. WASD move, scroll zoom.");
        ImGui::TextWrapped("Builder: 1-5 category, Q/E piece, C color, F Ground walk / Free cam, Hold RMB look, Del delete.");
        ImGui::TextWrapped("Ground walk: WASD walk, Shift run, Space jump.  Free cam: WASD fly, Space up, Ctrl down.");
        ImGui::TextWrapped("Esc: back (builder > map > menu > quit).");
    }
    if (ImGui::Button("Quit", ImVec2(-1, 34))) glfwSetWindowShouldClose(w, 1);
    ImGui::End();
}

static void drawMapUI(GLFWwindow* w) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Station Map", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextWrapped("Click a station on the map (or in the list) to select it, then press Build.");
    ImGui::TextDisabled("WASD: move map   Scroll: zoom   Esc: main menu");
    if (!g_msg.empty()) ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.4f, 1.0f), "%s", g_msg.c_str());
    ImGui::SeparatorText("Stations");
    for (int i = 0; i < (int)g_stations.size(); i++) {
        ImGui::PushID(i);
        if (ImGui::Selectable(g_stations[i].name.c_str(), g_mapSel == i)) { g_mapSel = i; g_msg.clear(); }
        ImGui::PopID();
    }
    ImGui::BeginDisabled();
    for (const Lot& l : LOCKED_LOTS) ImGui::Selectable(l.name, false);
    ImGui::EndDisabled();
    ImGui::End();

    if (g_mapSel >= 0 && g_mapSel < (int)g_stations.size()) { // detail card
        const Station& st = g_stations[g_mapSel];
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y - 20), ImGuiCond_Always, ImVec2(0.5f, 1.0f));
        ImGui::Begin("Selected station", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
        ImGui::Text("%s", st.name.c_str());
        ImGui::TextDisabled("Size %dx%d   Pieces: %d", (int)(st.x1 - st.x0), (int)(st.z1 - st.z0), (int)st.objects.size());
        if (ImGui::Button("Build here", ImVec2(220, 40))) enterStation(w, g_mapSel);
        ImGui::End();
    }
}

// Top bar: big category box + scrollable pieces, other categories below
static void drawHotbar(GLFWwindow* w) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(20, 10));
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x - 40, 0));
    ImGui::Begin("Hotbar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);
    ImGui::BeginChild("category", ImVec2(170, 66), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar);
    ImGui::SetWindowFontScale(1.6f);
    ImVec2 ts = ImGui::CalcTextSize(CAT_NAMES[g_cat]);
    ImGui::SetCursorPos(ImVec2((170 - ts.x) * 0.5f, (66 - ts.y) * 0.5f));
    ImGui::TextUnformatted(CAT_NAMES[g_cat]);
    ImGui::SetWindowFontScale(1.0f);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("pieces", ImVec2(0, 66), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
    if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f) ImGui::SetScrollX(ImGui::GetScrollX() - io.MouseWheel * 80.0f);
    for (int i = 0; i < (int)g_items[g_cat].size(); i++) {
        const Item& it = g_items[g_cat][i];
        bool sel = i == g_item;
        if (i > 0) ImGui::SameLine();
        ImGui::PushID(i);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(it.color.x * 0.55f, it.color.y * 0.55f, it.color.z * 0.55f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, sel ? ImVec4(1.0f, 0.85f, 0.3f, 1.0f) : ImVec4(0, 0, 0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, sel ? 2.5f : 0.0f);
        if (ImGui::Button(it.name.c_str(), ImVec2(110, 40))) selectItem(w, g_cat, i);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
        ImGui::PopID();
    }
    ImGui::EndChild();
    for (int c = 0; c < 5; c++) { // the other categories
        if (c == g_cat) continue;
        if (ImGui::SmallButton(CAT_NAMES[c])) selectItem(w, c, 0);
        ImGui::SameLine();
        ImGui::TextDisabled("-");
        ImGui::SameLine();
    }
    ImGui::TextDisabled("  * scroll for the next thing");
    ImGui::End();
}

// One first-person hand holding a tool. "Shine" (glow) shows which one is active.
static void drawHandArt(ImDrawList* dl, ImVec2 p, ImVec2 sz, bool left, bool active, bool hover) {
    auto P = [&](float fx, float fy) { return ImVec2(p.x + fx * sz.x, p.y + fy * sz.y); };
    auto L = [](ImVec2 a, ImVec2 b, float t) { return ImVec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t); };
    const ImU32 skin = IM_COL32(236, 190, 150, 255), skinDark = IM_COL32(180, 130, 100, 255);
    const ImU32 sleeve = IM_COL32(60, 70, 95, 255), outline = IM_COL32(35, 30, 30, 255);
    if (active || hover) {
        int a = active ? 24 : 10;
        for (int i = 8; i >= 1; i--) dl->AddCircleFilled(P(0.5f, 0.5f), sz.x * 0.06f * i + 10.0f, IM_COL32(255, 235, 140, a));
    }
    if (left) { // spade
        dl->AddQuadFilled(P(0.30f, 1.0f), P(0.72f, 1.0f), P(0.78f, 0.72f), P(0.38f, 0.66f), sleeve);
        dl->AddLine(P(0.20f, 0.28f), P(0.66f, 0.80f), IM_COL32(120, 80, 45, 255), sz.x * 0.06f);
        ImVec2 blade[5] = {P(0.04f, 0.10f), P(0.20f, 0.02f), P(0.34f, 0.14f), P(0.28f, 0.34f), P(0.10f, 0.30f)};
        dl->AddConvexPolyFilled(blade, 5, IM_COL32(170, 180, 190, 255));
        dl->AddPolyline(blade, 5, outline, ImDrawFlags_Closed, 2.0f);
        dl->AddRectFilled(P(0.36f, 0.50f), P(0.80f, 0.80f), skin, 14.0f);
        for (int i = 1; i <= 3; i++) dl->AddLine(P(0.36f + 0.11f * i, 0.52f), P(0.36f + 0.11f * i, 0.78f), skinDark, 2.0f);
        dl->AddRect(P(0.36f, 0.50f), P(0.80f, 0.80f), outline, 14.0f, 0, 2.0f);
    } else { // blueprint
        ImVec2 paper[4] = {P(0.22f, 0.10f), P(0.80f, 0.04f), P(0.86f, 0.50f), P(0.18f, 0.54f)};
        dl->AddConvexPolyFilled(paper, 4, IM_COL32(40, 90, 170, 255));
        for (int i = 1; i <= 3; i++) {
            float t = i / 4.0f;
            dl->AddLine(L(paper[0], paper[1], t), L(paper[3], paper[2], t), IM_COL32(255, 255, 255, 90), 1.0f);
            dl->AddLine(L(paper[0], paper[3], t), L(paper[1], paper[2], t), IM_COL32(255, 255, 255, 90), 1.0f);
        }
        dl->AddPolyline(paper, 4, IM_COL32(235, 240, 255, 255), ImDrawFlags_Closed, 2.0f);
        dl->AddQuadFilled(P(0.28f, 1.0f), P(0.70f, 1.0f), P(0.62f, 0.66f), P(0.22f, 0.72f), sleeve);
        dl->AddRectFilled(P(0.20f, 0.50f), P(0.64f, 0.80f), skin, 14.0f);
        for (int i = 1; i <= 3; i++) dl->AddLine(P(0.20f + 0.11f * i, 0.52f), P(0.20f + 0.11f * i, 0.78f), skinDark, 2.0f);
        dl->AddRect(P(0.20f, 0.50f), P(0.64f, 0.80f), outline, 14.0f, 0, 2.0f);
    }
    const char* label = left ? "Spade (remove)" : "Blueprint (build)";
    dl->AddText(ImVec2(p.x + 9, p.y + sz.y - 19), IM_COL32(0, 0, 0, 200), label);
    dl->AddText(ImVec2(p.x + 8, p.y + sz.y - 20), IM_COL32(255, 255, 255, 255), label);
}

static bool handButton(const char* id, ImVec2 pos, ImVec2 size, bool left, bool active) {
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin(id, nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
                                  ImGuiWindowFlags_NoSavedSettings);
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton("hand", size);
    drawHandArt(ImGui::GetWindowDrawList(), p, size, left, active, ImGui::IsItemHovered());
    ImGui::End();
    ImGui::PopStyleVar();
    return clicked;
}

// Small mouse diagram: left click = use, hold right = turn the camera
static void drawMouseHint(ImDrawList* dl, ImVec2 p) {
    const ImU32 line = IM_COL32(235, 235, 235, 230);
    dl->AddRect(p, ImVec2(p.x + 34, p.y + 54), line, 16.0f, 0, 2.0f);
    dl->AddLine(ImVec2(p.x + 17, p.y), ImVec2(p.x + 17, p.y + 22), line, 2.0f);
    dl->AddLine(ImVec2(p.x, p.y + 22), ImVec2(p.x + 34, p.y + 22), line, 2.0f);
    dl->AddRectFilled(ImVec2(p.x + 3, p.y + 3), ImVec2(p.x + 16, p.y + 21), IM_COL32(255, 235, 140, 140), 8.0f, ImDrawFlags_RoundCornersTopLeft);
    dl->AddText(ImVec2(p.x - 86, p.y + 8), line, "Click to use");
    dl->AddText(ImVec2(p.x + 44, p.y + 8), line, "Hold: turn camera");
}

static void drawBuilderUI(GLFWwindow* w) {
    ImGuiIO& io = ImGui::GetIO();
    drawHotbar(w);

    // Camera mode badge
    char cm[64];
    std::snprintf(cm, sizeof(cm), "%s   [F] switch", g_camMove == CAM_WALK ? "GROUND WALK" : "FREE CAM");
    ImVec2 cts = ImGui::CalcTextSize(cm);
    ImGui::GetBackgroundDrawList()->AddText(ImVec2((io.DisplaySize.x - cts.x) * 0.5f, io.DisplaySize.y - 30), IM_COL32(255, 235, 140, 255), cm);

    const ImVec2 hand(190, 230);
    if (handButton("Spade", ImVec2(10, io.DisplaySize.y - hand.y - 8), hand, true, g_mode == MODE_REMOVE)) setMode(w, MODE_REMOVE);
    if (handButton("Blueprint", ImVec2(io.DisplaySize.x - hand.x - 10, io.DisplaySize.y - hand.y - 8), hand, false, g_mode == MODE_BUILD)) setMode(w, MODE_BUILD);
    drawMouseHint(ImGui::GetBackgroundDrawList(), ImVec2(io.DisplaySize.x - hand.x - 190, io.DisplaySize.y - 80));

    ImGui::SetNextWindowPos(ImVec2(10, 170), ImGuiCond_FirstUseEver);
    ImGui::Begin("Tools", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("%s", g_stations[g_cur].name.c_str());
    if (ImGui::Button("< Back to map")) { exitToMap(w); ImGui::End(); return; }
    const char* names[] = {"Move", "Size", "Rotate", "Look"};
    const Mode modes[] = {MODE_MOVE, MODE_SIZE, MODE_ROTATE, MODE_FREECAM};
    for (int i = 0; i < 4; i++) {
        if (i > 0) ImGui::SameLine();
        if (ImGui::RadioButton(names[i], g_mode == modes[i])) setMode(w, modes[i]);
    }
    ImGui::SeparatorText("Camera");
    if (ImGui::RadioButton("Ground walk", g_camMove == CAM_WALK)) setCamMove(CAM_WALK);
    ImGui::SameLine();
    if (ImGui::RadioButton("Free cam", g_camMove == CAM_FLY)) setCamMove(CAM_FLY);
    ImGui::ColorEdit3("Color", &g_color.x);
    ImGui::TextDisabled("Hands: click one to use it");
    ImGui::TextDisabled("1-5: category   Q/E: piece   F: walk/fly");
    ImGui::TextDisabled("WASD move  Space jump/up  Ctrl down (fly)");
    ImGui::TextDisabled("Hold RMB: look   Tab: Look mode   Del: delete");
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 290, 170), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(270, 330), ImGuiCond_FirstUseEver);
    ImGui::Begin("Objects");
    ImGui::SeparatorText("Placed objects");
    ImGui::BeginChild("objlist", ImVec2(0, 140), ImGuiChildFlags_Borders);
    for (int i = 0; i < (int)g_objects.size(); i++) {
        char label[96];
        std::snprintf(label, sizeof(label), "%d: %s", i + 1, g_shapeNames[g_objects[i].shape].c_str());
        ImGui::PushID(i);
        if (ImGui::Selectable(label, g_selected == i)) g_selected = i;
        ImGui::PopID();
    }
    ImGui::EndChild();
    if (g_selected >= 0 && g_selected < (int)g_objects.size()) {
        Object& o = g_objects[g_selected];
        ImGui::SeparatorText("Selected");
        ImGui::DragFloat3("Position", &o.pos.x, 0.05f);
        ImGui::DragFloat3("Scale", &o.scale.x, 0.05f, MIN_SCALE, 50.0f);
        ImGui::ColorEdit3("Tint", &o.color.x);
        if (ImGui::Button("Reset rotation")) o.rot = Mat4::identity();
        ImGui::SameLine();
        if (ImGui::Button("Delete")) deleteSelected();
    }
    ImGui::End();
}

static void drawUI(GLFWwindow* w) {
    if (g_scene == SCENE_MENU) drawMenuUI(w);
    else if (g_scene == SCENE_MAP) drawMapUI(w);
    else drawBuilderUI(w);
}

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Station Design", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "Failed to init GLEW\n");
        return 1;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true); // chains the callbacks set above
    ImGui_ImplOpenGL3_Init("#version 330");

    GLuint program = makeProgram();
    GLint uModel = glGetUniformLocation(program, "uModel");
    GLint uView = glGetUniformLocation(program, "uView");
    GLint uProj = glGetUniformLocation(program, "uProj");
    GLint uColor = glGetUniformLocation(program, "uColor");
    GLint uLightPos = glGetUniformLocation(program, "uLightPos");
    GLint uCamPos = glGetUniformLocation(program, "uCamPos");
    GLint uChecker = glGetUniformLocation(program, "uChecker");
    GLint uAlpha = glGetUniformLocation(program, "uAlpha");

    Mesh plane = makePlane();
    Mesh cone = makeCone();
    Mesh torus = makeTorus();
    g_shapes.push_back(makeCube());
    g_shapeNames.push_back("Square");
    g_shapes.push_back(makeCylinder());
    g_shapeNames.push_back("Circle");
    loadModels();
    buildItems();

    { // one selectable station for now
        Station st;
        st.name = "Central Station";
        st.x0 = -10; st.x1 = 10; st.z0 = -8; st.z1 = 8;
        st.color = PALETTE[2];
        g_stations.push_back(st);
    }

    double last = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = (float)(now - last);
        last = now;

        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawUI(window);
        processInput(window, dt);
        ImGuiIO& io = ImGui::GetIO();
        bool overUI = io.WantCaptureMouse;

        if (g_scene == SCENE_MENU) { // slowly orbiting camera behind the menu
            float a = (float)now * 0.15f;
            Vec3 eye(std::cos(a) * 30.0f, 16.0f, std::sin(a) * 30.0f), d = normalize(Vec3(0, 0, 0) - eye);
            g_camera.position = eye;
            g_camera.yaw = std::atan2(d.z, d.x) * 180.0f / 3.14159265f;
            g_camera.pitch = std::asin(d.y) * 180.0f / 3.14159265f;
            g_camera.fov = 60.0f;
        } else if (g_scene == SCENE_MAP) { // map camera: straight down
            g_camera.position = Vec3(g_mapX, g_mapH, g_mapZ);
            g_camera.yaw = -90.0f;
            g_camera.pitch = -89.0f;
            g_camera.fov = 60.0f;
        }
        Ray ray = cursorRay(window);
        bool lmbDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        Vec3 place;
        bool hasPlace = false;

        if (g_scene == SCENE_MENU) {
            // nothing: only the menu UI
        } else if (g_scene == SCENE_MAP) {
            // ----- Map: select a station (or drag out a lot if enabled) -----
            int cx, cz;
            bool onGround = groundCell(ray, cx, cz);
            g_hoverStation = -1;
            if (onGround && !overUI)
                for (int i = 0; i < (int)g_stations.size(); i++) {
                    const Station& st = g_stations[i];
                    if (cx >= st.x0 && cx < st.x1 && cz >= st.z0 && cz < st.z1) g_hoverStation = i;
                }
            if (g_lmbPressed && onGround) {
                if (g_hoverStation >= 0) { g_mapSel = g_hoverStation; g_msg.clear(); }
                else if (lockedAt(cx, cz)) { g_mapSel = -1; g_msg = "That station is locked for now."; }
                else if (ALLOW_NEW_STATIONS) {
                    g_areaDrag = true;
                    g_ax0 = g_ax1 = cx; g_az0 = g_az1 = cz;
                    g_msg.clear();
                } else g_mapSel = -1;
            }
            if (g_areaDrag) {
                if (onGround) { g_ax1 = cx; g_az1 = cz; }
                if (!lmbDown) { g_areaDrag = false; finishArea(window); }
            }
        } else {
            // ----- Builder -----
            // Look mode: the cursor stays visible; mouse movement turns the camera,
            // and holding the cursor at a window edge keeps turning that way.
            if (g_mode == MODE_FREECAM && !overUI) {
                double mx, my;
                int ww, wh;
                glfwGetCursorPos(window, &mx, &my);
                glfwGetWindowSize(window, &ww, &wh);
                const double margin = 24.0, edgeSpeed = 700.0;
                float ex = mx < margin ? -1.0f : (mx > ww - margin ? 1.0f : 0.0f);
                float ey = my < margin ? -1.0f : (my > wh - margin ? 1.0f : 0.0f);
                g_camera.look(ex * (float)edgeSpeed * dt, ey * (float)edgeSpeed * dt);
            }

            bool editing = g_mode != MODE_FREECAM && !g_rmbLook && !overUI;
            Hit hit;
            g_hover = -1;
            if (!lmbDown) g_drag.active = false;
            if (g_drag.active && g_selected >= 0) updateDrag(ray);
            else if (editing) {
                if (g_mode == MODE_BUILD) hit = raycastScene(ray);
                else g_hover = raycastScene(ray).object; // the object under the cursor shines
            }
            const Item& item = g_items[g_cat][g_item];
            hasPlace = editing && g_mode == MODE_BUILD && placement(hit, ray, item, place);

            if (g_lmbPressed && editing) {
                if (g_mode == MODE_BUILD) {
                    if (hasPlace) {
                        g_objects.push_back({item.shape, place, item.scale, g_color});
                        g_selected = (int)g_objects.size() - 1;
                    }
                } else if (g_mode == MODE_REMOVE) {
                    deleteObject(g_hover);
                } else {
                    int axis = pickGizmo(ray);
                    if (axis >= 0) beginDrag(ray, axis);
                    else g_selected = raycastScene(ray).object; // click empty space / ground = deselect
                }
            }
        }
        g_lmbPressed = false;

        // ----- Render -----
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        if (h == 0) h = 1;
        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.07f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Mat4 view = g_camera.view();
        Mat4 proj = g_camera.projection((float)w / (float)h);

        glUseProgram(program);
        glUniformMatrix4fv(uView, 1, GL_FALSE, view.m);
        glUniformMatrix4fv(uProj, 1, GL_FALSE, proj.m);
        glUniform3f(uLightPos, 6.0f, 10.0f, 6.0f);
        glUniform3f(uCamPos, g_camera.position.x, g_camera.position.y, g_camera.position.z);

        auto drawMesh = [&](const Mesh& m, const Mat4& model, const Vec3& c, bool checker, float alpha = 1.0f) {
            glUniformMatrix4fv(uModel, 1, GL_FALSE, model.m);
            glUniform3f(uColor, c.x, c.y, c.z);
            glUniform1i(uChecker, checker ? 1 : 0);
            glUniform1f(uAlpha, alpha);
            m.draw();
        };
        auto drawObject = [&](const Object& o, const Vec3& c) {
            drawMesh(g_shapes[o.shape], translate(o.pos) * o.rot * scale(o.scale), c, false);
        };
        auto drawLot = [&](float x0, float z0, float x1, float z1, const Vec3& c, float alpha) { // flat marker on the floor
            drawMesh(g_shapes[0], translate(Vec3((x0 + x1) * 0.5f, 0.03f, (z0 + z1) * 0.5f)) * scale(Vec3(x1 - x0, 0.04f, z1 - z0)), c, false, alpha);
        };
        auto drawBorder = [&](float x0, float z0, float x1, float z1, const Vec3& c) {
            const float t = 0.12f;
            Vec3 s1(x1 - x0 + t, 0.1f, t), s2(t, 0.1f, z1 - z0 + t);
            drawMesh(g_shapes[0], translate(Vec3((x0 + x1) * 0.5f, 0.08f, z0)) * scale(s1), c, false);
            drawMesh(g_shapes[0], translate(Vec3((x0 + x1) * 0.5f, 0.08f, z1)) * scale(s1), c, false);
            drawMesh(g_shapes[0], translate(Vec3(x0, 0.08f, (z0 + z1) * 0.5f)) * scale(s2), c, false);
            drawMesh(g_shapes[0], translate(Vec3(x1, 0.08f, (z0 + z1) * 0.5f)) * scale(s2), c, false);
        };

        drawMesh(plane, Mat4::identity(), Vec3(1, 1, 1), true);
        for (const Station& st : g_stations)
            for (const Object& o : st.objects) drawObject(o, o.color);
        for (int i = 0; i < (int)g_objects.size(); i++) {
            const Object& o = g_objects[i];
            Vec3 c = o.color;
            if (i == g_selected) c = c * 0.65f + Vec3(0.35f, 0.35f, 0.35f);       // selected
            else if (i == g_hover) c = c * 0.5f + Vec3(0.5f, 0.42f, 0.1f);        // shine: this is the one you would click
            drawObject(o, c);
        }
        for (int i = 0; i < (int)g_stations.size(); i++) {
            const Station& st = g_stations[i];
            bool inMap = g_scene != SCENE_BUILDER;
            bool sel = inMap && i == g_mapSel, hov = inMap && i == g_hoverStation;
            float a = inMap ? (sel ? 0.85f : (hov ? 0.75f : 0.55f)) : (i == g_cur ? 0.25f : 0.5f);
            drawLot(st.x0, st.z0, st.x1, st.z1, st.color, a);
            drawBorder(st.x0, st.z0, st.x1, st.z1, (sel || i == g_cur) ? Vec3(1.0f, 0.9f, 0.3f) : Vec3(0.9f, 0.9f, 0.9f));
        }
        if (g_scene != SCENE_BUILDER)
            for (const Lot& l : LOCKED_LOTS) {
                drawLot(l.x0, l.z0, l.x1, l.z1, Vec3(0.35f, 0.35f, 0.38f), 0.45f);
                drawBorder(l.x0, l.z0, l.x1, l.z1, Vec3(0.2f, 0.2f, 0.22f));
            }
        if (g_areaDrag) {
            float x0 = (float)std::min(g_ax0, g_ax1), x1 = (float)std::max(g_ax0, g_ax1) + 1;
            float z0 = (float)std::min(g_az0, g_az1), z1 = (float)std::max(g_az0, g_az1) + 1;
            drawLot(x0, z0, x1, z1, Vec3(0.3f, 0.9f, 0.4f), 0.55f);
        }
        if (hasPlace) { // see-through preview of the piece about to be placed
            const Item& item = g_items[g_cat][g_item];
            drawMesh(g_shapes[item.shape], translate(place) * scale(item.scale), g_color, false, 0.5f);
        }

        if (g_scene == SCENE_BUILDER && g_selected >= 0 && (g_mode == MODE_MOVE || g_mode == MODE_SIZE || g_mode == MODE_ROTATE)) {
            const Object& o = g_objects[g_selected];
            float k = gizmoK();
            glDisable(GL_DEPTH_TEST); // handles are always visible, even inside the object
            for (int a = 0; a < GIZMO_AXES; a++) {
                Mat4 axisRot = Mat4::identity(); // turns local +Y into the chosen axis
                if (a == 0) axisRot = rotateZ(radians(-90.0f));
                if (a == 2) axisRot = rotateX(radians(90.0f));
                if (g_mode == MODE_ROTATE) {
                    float r = ringRadius(o, k);
                    drawMesh(torus, translate(o.pos) * axisRot * scale(Vec3(r, r, r)), AXIS_COLOR[a], false);
                } else {
                    float len = gizmoLen(o, a, k), shaft = len - 0.3f * k;
                    Mat4 base = translate(o.pos) * gizmoFrame(o) * axisRot;
                    drawMesh(g_shapes[1], base * translate(Vec3(0, shaft * 0.5f, 0)) * scale(Vec3(0.05f * k, shaft, 0.05f * k)), AXIS_COLOR[a], false);
                    drawMesh(cone, base * translate(Vec3(0, len - 0.15f * k, 0)) * scale(Vec3(0.2f * k, 0.3f * k, 0.2f * k)), AXIS_COLOR[a], false);
                }
            }
            glEnable(GL_DEPTH_TEST);
        }

        if (g_scene == SCENE_MAP) { // station names on the map
            auto label = [&](float x, float z, const char* text, ImU32 col) {
                ImVec2 sp;
                if (!projectToScreen(Vec3(x, 0.1f, z), view, proj, io.DisplaySize, sp)) return;
                ImVec2 ts = ImGui::CalcTextSize(text);
                ImGui::GetBackgroundDrawList()->AddText(ImVec2(sp.x - ts.x * 0.5f + 1, sp.y - ts.y * 0.5f + 1), IM_COL32(0, 0, 0, 220), text);
                ImGui::GetBackgroundDrawList()->AddText(ImVec2(sp.x - ts.x * 0.5f, sp.y - ts.y * 0.5f), col, text);
            };
            for (const Station& st : g_stations)
                label((st.x0 + st.x1) * 0.5f, (st.z0 + st.z1) * 0.5f, st.name.c_str(), IM_COL32(255, 255, 255, 255));
            for (const Lot& l : LOCKED_LOTS)
                label((l.x0 + l.x1) * 0.5f, (l.z0 + l.z1) * 0.5f, l.name, IM_COL32(170, 170, 175, 255));
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    plane.destroy();
    cone.destroy();
    torus.destroy();
    for (Mesh& m : g_shapes) m.destroy();
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}