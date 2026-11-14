#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// Matriks 4x4 column-major
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
    float yaw = -90.0f; // -90 = menghadap -Z
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

    // Vertex berformat [posisi(3), normal(3)]
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
            v.insert(v.end(), {x, y, z, x, y, z}); // normal = posisi
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
        std::fprintf(stderr, "Gagal init GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Mini 3D Engine", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Gagal membuat window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "Gagal init GLEW\n");
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

    std::vector<Entity> scene = {
        {&plane, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, true},
        {&sphere, {0, 1.5f, 0}, {1.5f, 1.5f, 1.5f}, {0.9f, 0.25f, 0.2f}}, // bola utama
        {&sphere, {-4, 1.0f, -2}, {1.0f, 1.0f, 1.0f}, {0.2f, 0.6f, 0.95f}},
        {&sphere, {4, 0.7f, -1}, {0.7f, 0.7f, 0.7f}, {0.3f, 0.85f, 0.4f}},
        {&sphere, {1.5f, 0.4f, 3}, {0.4f, 0.4f, 0.4f}, {0.95f, 0.8f, 0.2f}},
    };

    double last = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = (float)(now - last);
        last = now;

        glfwPollEvents();
        processInput(window, dt);

        scene[1].position.y = 1.5f + 0.5f * std::sin((float)now * 1.5f); // bola utama naik-turun

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

        for (const Entity& e : scene) {
            Mat4 model = translate(e.position) * scale(e.scale);
            glUniformMatrix4fv(uModel, 1, GL_FALSE, model.m);
            glUniform3f(uColor, e.color.x, e.color.y, e.color.z);
            glUniform1i(uChecker, e.checker ? 1 : 0);
            e.mesh->draw();
        }

        glfwSwapBuffers(window);
    }

    sphere.destroy();
    plane.destroy();
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}