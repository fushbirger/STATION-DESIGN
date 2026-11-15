#include <GL/glew.h>
#include <GLFW/glfw3.h>

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

struct Entity {
    const Mesh* mesh;
    Vec3 position;
    Vec3 scale;
    Vec3 color;
    bool checker = false;
};

static Camera g_camera;
static bool g_firstMouse = true;
static double g_lastX = 0, g_lastY = 0;


// ---------- Block system ----------
struct Block { int shape; int x, y, z; Vec3 color; }; // shape = index into g_shapes
static std::vector<Block> g_blocks;
static std::vector<Mesh> g_shapes;           // 0 = square, 1 = circle, 2+ = loaded models
static std::vector<std::string> g_shapeNames;
static int g_shape = 0, g_colorIdx = 0;
static bool g_clickPlace = false, g_clickRemove = false;
static const Vec3 PALETTE[] = {{0.85f, 0.85f, 0.85f}, {0.9f, 0.25f, 0.2f}, {0.2f, 0.6f, 0.95f},
                               {0.3f, 0.85f, 0.4f}, {0.95f, 0.8f, 0.2f}, {0.6f, 0.4f, 0.25f}};
static const int PALETTE_N = 6;


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
    fs::path dir = fs::path(ASSET_DIR) / "model";
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        std::fprintf(stderr, "Model folder not found: %s\n", dir.string().c_str());
        return;
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (entry.is_regular_file() && ext == ".obj") files.push_back(entry.path());
    }
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

static int findBlock(int x, int y, int z) {
    for (size_t i = 0; i < g_blocks.size(); i++)
        if (g_blocks[i].x == x && g_blocks[i].y == y && g_blocks[i].z == z) return (int)i;
    return -1;
}

struct Target { bool canPlace = false; int px = 0, py = 0, pz = 0; int hitBlock = -1; };

// Cast a ray from the screen center: find the block hit and the empty cell in front of it
static Target raycast() {
    Target t;
    Vec3 o = g_camera.position, d = g_camera.front();
    int lx = 0, ly = 0, lz = 0;
    bool havePrev = false;
    for (float s = 0; s < 20.0f; s += 0.02f) {
        Vec3 p = o + d * s;
        if (p.y < 0) break; // hit the floor
        int cx = (int)std::floor(p.x), cy = (int)std::floor(p.y), cz = (int)std::floor(p.z);
        if (havePrev && cx == lx && cy == ly && cz == lz) continue;
        int b = findBlock(cx, cy, cz);
        if (b >= 0) { t.hitBlock = b; break; }
        lx = cx; ly = cy; lz = cz; havePrev = true;
    }
    if (havePrev) { t.canPlace = true; t.px = lx; t.py = ly; t.pz = lz; }
    return t;
}

static void updateTitle(GLFWwindow* w) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "Station Design | Shape: %s | Blocks: %d",
                  g_shapeNames[g_shape].c_str(), (int)g_blocks.size());
    glfwSetWindowTitle(w, buf);
}

static void mouseButtonCallback(GLFWwindow*, int button, int action, int) {
    if (action != GLFW_PRESS) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) g_clickPlace = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) g_clickRemove = true;
}

static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    int n = (int)g_shapes.size();
    if (key == GLFW_KEY_1) g_shape = 0;
    if (key == GLFW_KEY_2) g_shape = 1;
    if (key == GLFW_KEY_E) g_shape = (g_shape + 1) % n;     // next shape / model
    if (key == GLFW_KEY_Q) g_shape = (g_shape + n - 1) % n; // previous shape / model
    if (key == GLFW_KEY_C) g_colorIdx = (g_colorIdx + 1) % PALETTE_N;
    updateTitle(w);
}

static void mouseCallback(GLFWwindow*, double x, double y) {
    if (g_firstMouse) { g_lastX = x; g_lastY = y; g_firstMouse = false; }
    g_camera.look((float)(x - g_lastX), (float)(y - g_lastY));
    g_lastX = x;
    g_lastY = y;
}

static void scrollCallback(GLFWwindow*, double, double yoff) {
    g_camera.fov -= (float)yoff * 2.0f;
    if (g_camera.fov < 20.0f) g_camera.fov = 20.0f;
    if (g_camera.fov > 100.0f) g_camera.fov = 100.0f;
}

static void processInput(GLFWwindow* w, float dt) {
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);

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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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
    g_shapes.push_back(makeCube());
    g_shapeNames.push_back("Square");
    g_shapes.push_back(makeCylinder());
    g_shapeNames.push_back("Circle");
    loadModels();
    Entity ground = {&plane, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, true};
    updateTitle(window);

    double last = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = (float)(now - last);
        last = now;

        glfwPollEvents();
        processInput(window, dt);

        Target tg = raycast();
        if (g_clickPlace && tg.canPlace && std::fabs(tg.px) < 50 && std::fabs(tg.pz) < 50)
            g_blocks.push_back({g_shape, tg.px, tg.py, tg.pz, PALETTE[g_colorIdx]});
        if (g_clickRemove && tg.hitBlock >= 0) g_blocks.erase(g_blocks.begin() + tg.hitBlock);
        if (g_clickPlace || g_clickRemove) updateTitle(window);
        g_clickPlace = g_clickRemove = false;

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

        auto drawEntity = [&](const Entity& e) {
            Mat4 model = translate(e.position) * scale(e.scale);
            glUniformMatrix4fv(uModel, 1, GL_FALSE, model.m);
            glUniform3f(uColor, e.color.x, e.color.y, e.color.z);
            glUniform1i(uChecker, e.checker ? 1 : 0);
            e.mesh->draw();
        };
        drawEntity(ground);
        for (const Block& bl : g_blocks)
            drawEntity({&g_shapes[bl.shape], {bl.x + 0.5f, bl.y + 0.5f, bl.z + 0.5f}, {1, 1, 1}, bl.color});
        if (tg.canPlace) // marker for the cell where the next block will be placed
            drawEntity({&sphere, {tg.px + 0.5f, tg.py + 0.5f, tg.pz + 0.5f}, {0.12f, 0.12f, 0.12f}, PALETTE[g_colorIdx]});

        glfwSwapBuffers(window);
    }

    sphere.destroy();
    plane.destroy();
    for (Mesh& m : g_shapes) m.destroy();
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}