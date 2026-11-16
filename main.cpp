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

static Mesh makeSphere(int stacks = 32, int sectors = 64) {
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
    FragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
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


//   Scene objects  
struct Object { int shape; Vec3 pos, scale, color; Mat4 rot = Mat4::identity(); }; // shape = index into g_shapes
static std::vector<Object> g_objects;
static int g_selected = -1;                   // index into g_objects, -1 = none
static std::vector<Mesh> g_shapes;            // 0 = square, 1 = circle, 2+ = loaded models
static std::vector<std::string> g_shapeNames;
static int g_shape = 0;                       // shape used when placing
static const Vec3 PALETTE[] = {{0.85f, 0.85f, 0.85f}, {0.9f, 0.25f, 0.2f}, {0.2f, 0.6f, 0.95f},
                               {0.3f, 0.85f, 0.4f}, {0.95f, 0.8f, 0.2f}, {0.6f, 0.4f, 0.25f}};
static const int PALETTE_N = 6;
static int g_colorIdx = 0;
static Vec3 g_color = PALETTE[0];             // color used when placing

enum Mode { MODE_BUILD, MODE_MOVE, MODE_SIZE, MODE_ROTATE, MODE_FREECAM };
static Mode g_mode = MODE_BUILD, g_lastTool = MODE_BUILD;
static bool g_lmbPressed = false;             // left click this frame (not over the UI)
static bool g_rmbLook = false;                // right mouse held: look around

static const int GIZMO_AXES = 3;              // 3 = X, Y and Z handles (2 = only X and Y)
static const float GRID_SNAP = 0.25f;         // move / size step
static const float MIN_SCALE = 0.25f;
static const float ROT_SNAP = 3.14159265f / 12.0f; // 15 degrees


//   Model loading (Wavefront .obj)  
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

//   Picking  
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

// Grid cell (as a center point) next to the face that was hit. False if outside the floor.
static bool hitCell(const Hit& h, const Ray& ray, Vec3& center) {
    if (!h.hit) return false;
    Vec3 p = ray.o + ray.d * h.t + h.normal * 0.01f;
    int cx = (int)std::floor(p.x), cy = (int)std::floor(p.y), cz = (int)std::floor(p.z);
    if (cx < -50 || cx >= 50 || cz < -50 || cz >= 50 || cy < 0) return false;
    center = Vec3(cx + 0.5f, cy + 0.5f, cz + 0.5f);
    return true;
}

//   Gizmo (arrows for Move / Size, rings for Rotate)  
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

//   Mode / selection  
static void setMode(GLFWwindow* w, Mode m) {
    if (m != MODE_FREECAM) g_lastTool = m;
    g_mode = m;
    g_drag.active = false;
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // the cursor stays visible in every mode
    g_firstMouse = true;
}

static void deleteSelected() {
    if (g_selected < 0 || g_selected >= (int)g_objects.size()) return;
    g_objects.erase(g_objects.begin() + g_selected);
    g_selected = -1;
    g_drag.active = false;
}

//   Input callbacks  
static void mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
    bool overUI = ImGui::GetIO().WantCaptureMouse;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !overUI) g_lmbPressed = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS && g_mode != MODE_FREECAM && !overUI) { // hold RMB to look around
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
    int n = (int)g_shapes.size();
    if (key == GLFW_KEY_1) g_shape = 0;
    if (key == GLFW_KEY_2) g_shape = 1;
    if (key == GLFW_KEY_E) g_shape = (g_shape + 1) % n;     // next shape / model
    if (key == GLFW_KEY_Q) g_shape = (g_shape + n - 1) % n; // previous shape / model
    if (key == GLFW_KEY_C) { g_colorIdx = (g_colorIdx + 1) % PALETTE_N; g_color = PALETTE[g_colorIdx]; }
    if (key == GLFW_KEY_TAB) setMode(w, g_mode == MODE_FREECAM ? g_lastTool : MODE_FREECAM);
    if (key == GLFW_KEY_DELETE) deleteSelected();
}

static void mouseCallback(GLFWwindow*, double x, double y) {
    if (g_firstMouse) { g_lastX = x; g_lastY = y; g_firstMouse = false; }
    bool freeLook = g_mode == MODE_FREECAM && !ImGui::GetIO().WantCaptureMouse; // not while over the UI
    if (freeLook || g_rmbLook) g_camera.look((float)(x - g_lastX), (float)(y - g_lastY));
    g_lastX = x;
    g_lastY = y;
}

static void scrollCallback(GLFWwindow*, double, double yoff) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    g_camera.fov -= (float)yoff * 2.0f;
    if (g_camera.fov < 20.0f) g_camera.fov = 20.0f;
    if (g_camera.fov > 100.0f) g_camera.fov = 100.0f;
}

static void processInput(GLFWwindow* w, float dt) {
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    if (ImGui::GetIO().WantTextInput) return; // typing into a UI field

    float v = g_camera.speed * dt;
    if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) v *= 3.0f;

    Vec3 f = g_camera.front();
    Vec3 r = g_camera.right();
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) g_camera.position += f * v;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) g_camera.position -= f * v;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) g_camera.position += r * v;
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) g_camera.position -= r * v;
    if (glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS) g_camera.position += Vec3(0, 1, 0) * v;
    if (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) g_camera.position -= Vec3(0, 1, 0) * v;
}

//   UI  
static void drawUI(GLFWwindow* w) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Tools", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    const char* names[] = {"Build", "Move", "Size", "Rotate", "Free Cam"};
    for (int i = 0; i < 5; i++) {
        if (i > 0) ImGui::SameLine();
        if (ImGui::RadioButton(names[i], (int)g_mode == i)) setMode(w, (Mode)i);
    }
    ImGui::ColorEdit3("Color", &g_color.x);
    ImGui::TextDisabled("WASD/Space/Ctrl: fly   Hold RMB: look");
    ImGui::TextDisabled("Free Cam: just move the mouse to look");
    ImGui::TextDisabled("Tab: Free Cam on/off   Del: delete");
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(10, 130), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(270, 430), ImGuiCond_FirstUseEver);
    ImGui::Begin("Objects");
    ImGui::SeparatorText("Shape to place");
    for (int i = 0; i < (int)g_shapes.size(); i++) {
        ImGui::PushID(i);
        if (ImGui::Selectable(g_shapeNames[i].c_str(), g_shape == i)) g_shape = i;
        ImGui::PopID();
    }
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

    Mesh sphere = makeSphere();
    Mesh plane = makePlane();
    Mesh cone = makeCone();
    Mesh torus = makeTorus();
    g_shapes.push_back(makeCube());
    g_shapeNames.push_back("Square");
    g_shapes.push_back(makeCylinder());
    g_shapeNames.push_back("Circle");
    loadModels();

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

        // Free Cam: the cursor stays visible; mouse movement turns the camera,
        // and holding the cursor at a window edge keeps turning that way.
        if (g_mode == MODE_FREECAM && !ImGui::GetIO().WantCaptureMouse) {
            double mx, my;
            int ww, wh;
            glfwGetCursorPos(window, &mx, &my);
            glfwGetWindowSize(window, &ww, &wh);
            const double margin = 24.0, edgeSpeed = 700.0; // pixels-equivalent per second
            float ex = mx < margin ? -1.0f : (mx > ww - margin ? 1.0f : 0.0f);
            float ey = my < margin ? -1.0f : (my > wh - margin ? 1.0f : 0.0f);
            g_camera.look(ex * (float)edgeSpeed * dt, ey * (float)edgeSpeed * dt);
        }

        // ----- World interaction -----
        bool editing = g_mode != MODE_FREECAM && !g_rmbLook && !ImGui::GetIO().WantCaptureMouse;
        Ray ray = cursorRay(window);
        Hit hit;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) g_drag.active = false;
        if (g_drag.active && g_selected >= 0) updateDrag(ray);
        else if (editing && g_mode == MODE_BUILD) hit = raycastScene(ray);

        Vec3 cell;
        bool hasCell = editing && g_mode == MODE_BUILD && hitCell(hit, ray, cell);
        if (g_lmbPressed && editing) {
            if (g_mode == MODE_BUILD) {
                if (hasCell) {
                    g_objects.push_back({g_shape, cell, Vec3(1, 1, 1), g_color});
                    g_selected = (int)g_objects.size() - 1;
                }
            } else {
                int axis = pickGizmo(ray);
                if (axis >= 0) beginDrag(ray, axis);
                else g_selected = raycastScene(ray).object; // click empty space / ground = deselect
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

        auto drawMesh = [&](const Mesh& m, const Mat4& model, const Vec3& c, bool checker) {
            glUniformMatrix4fv(uModel, 1, GL_FALSE, model.m);
            glUniform3f(uColor, c.x, c.y, c.z);
            glUniform1i(uChecker, checker ? 1 : 0);
            m.draw();
        };

        drawMesh(plane, Mat4::identity(), Vec3(1, 1, 1), true);
        for (int i = 0; i < (int)g_objects.size(); i++) {
            const Object& o = g_objects[i];
            Vec3 c = i == g_selected ? o.color * 0.65f + Vec3(0.35f, 0.35f, 0.35f) : o.color; // highlight selection
            drawMesh(g_shapes[o.shape], translate(o.pos) * o.rot * scale(o.scale), c, false);
        }
        if (hasCell) // marker for the cell where the next object will be placed
            drawMesh(sphere, translate(cell) * scale(Vec3(0.12f, 0.12f, 0.12f)), g_color, false);

        if (g_selected >= 0 && g_mode != MODE_BUILD && g_mode != MODE_FREECAM) {
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

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    sphere.destroy();
    plane.destroy();
    cone.destroy();
    torus.destroy();
    for (Mesh& m : g_shapes) m.destroy();
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}