#include <thread>
#include <future>
#include <queue>

#include <pangolin/pangolin.h>
#include <pangolin/geometry/geometry.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>

#include <pangolin/utils/file_utils.h>

#include <pangolin/geometry/geometry_ply.h>
#include <pangolin/geometry/glgeometry.h>

#include <pangolin/utils/argagg.hpp>

#include "TextureShader.h"
#include "util.h"

#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

int main(int argc, char **argv) {
    const float w = 640.0f;
    const float h = 480.0f;
    const float f = 300.0f;

    using namespace pangolin;

    argagg::parser argparser{{
                                     {"help", {"-h", "--help"}, "Print usage information and exit.", 0},
                                     {"model", {"-m", "--model", "--mesh"}, "3D Model to load (obj or ply)", 1},
                                     {"matcap", {"--matcap"}, "Matcap (material capture) images to load for shading",
                                      1},
                                     {"envmap", {"--envmap", "-e"}, "Equirect environment map for skybox", 1},
                                     {"mode", {"--mode"},
                                      "Render mode to use {show_uv, show_texture, show_color, show_normal, show_matcap, show_vertex}",
                                      1},
                                     {"bounds", {"--aabb"}, "Show axis-aligned bounding-box", 0},
                                     {"show_axis", {"--axis"}, "Show axis coordinates for Origin", 0},
                                     {"show_x0", {"--x0"}, "Show X=0 Plane", 0},
                                     {"show_y0", {"--y0"}, "Show Y=0 Plane", 0},
                                     {"show_z0", {"--z0"}, "Show Z=0 Plane", 0},
                                     {"cull_backfaces", {"--cull"}, "Enable backface culling", 0},
                                     {"spin", {"--spin"},
                                      "Spin models around an axis {none, negx, x, negy, y, negz, z}", 1},
                             }};

    argagg::parser_results args = argparser.parse(argc, argv);
    if ((bool) args["help"] || !args.has_option("model")) {
        std::cerr << "usage: ModelViewer [options]" << std::endl
                  << argparser << std::endl;
        return 0;
    }

    // Options
    bool show_bounds = args.has_option("bounds");
    bool show_axis = args.has_option("show_axis");
    bool show_x0 = args.has_option("show_x0");
    bool show_y0 = args.has_option("show_y0");
    bool show_z0 = args.has_option("show_z0");
    bool cull_backfaces = args.has_option("cull_backfaces");
    // Create Window for rendering
    pangolin::CreateWindowAndBind("Main", w, h);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(w, h, f, f, w / 2.0, h / 2.0, 0.1, 1000),
            pangolin::ModelViewLookAt(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, pangolin::AxisY)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -w / h)
            .SetHandler(&handler);

    // Load Geometry asynchronously
    const pangolin::Geometry geom_to_load = pangolin::LoadGeometry(ExpandGlobOption(args["model"])[0]);
    auto aabb = pangolin::GetAxisAlignedBox(geom_to_load);
    Eigen::AlignedBox3f total_aabb;
    total_aabb.extend(aabb);
    const Eigen::Vector3f center = total_aabb.center();
    const Eigen::Vector3f view = center + Eigen::Vector3f(1.2, 0.8, 1.2) *
                                          std::max((total_aabb.max() - center).norm(),
                                                   (center - total_aabb.min()).norm());
    const auto mvm = pangolin::ModelViewLookAt(view[0], view[1], view[2], center[0], center[1], center[2],
                                               pangolin::AxisY);
    const double far = 100.0 * (total_aabb.max() - total_aabb.min()).norm();
    const double near = far / 1e6;
    const auto proj = pangolin::ProjectionMatrix(w, h, f, f, w / 2.0, h / 2.0, near, far);
    s_cam.SetModelViewMatrix(mvm);
    s_cam.SetProjectionMatrix(proj);
    const pangolin::GlGeometry geomToRender = pangolin::ToGlGeometry(geom_to_load);
    // Render tree for holding object position
    pangolin::AxisDirection spin_other = pangolin::AxisNone;
    pangolin::GlSlProgram default_prog;
    auto LoadProgram = [&]() {
        default_prog.ClearShaders();
        default_prog.AddShader(pangolin::GlSlAnnotatedShader, shader);
        default_prog.Link();
    };
    LoadProgram();
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });

    // Show axis and axis planes
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });

    Eigen::Vector3d Pick_w = handler.Selected_P_w();
    std::vector<Eigen::Vector3d> Picks_w;
    cv::Mat Twc;
    default_prog.Bind();

    while (!pangolin::ShouldQuit()) {
        if ((handler.Selected_P_w() - Pick_w).norm() > 1E-6) {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                      << std::endl;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Load any pending geometry to the GPU.
        if (d_cam.IsShown()) {
            d_cam.Activate();

            if (cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }
            default_prog.SetUniform("KT_cw",  s_cam.GetProjectionMatrix() *  s_cam.GetModelViewMatrix());
            pangolin::GlDraw( default_prog, geomToRender );

            s_cam.Apply();
            if (show_x0) pangolin::glDraw_x0(10.0, 10);
            if (show_y0) pangolin::glDraw_y0(10.0, 10);
            if (show_z0) pangolin::glDraw_z0(10.0, 10);
            if (show_axis) pangolin::glDrawAxis(10.0);
            if (show_bounds) pangolin::glDrawAlignedBox(total_aabb);

            glDisable(GL_CULL_FACE);
        }

        pangolin::FinishFrame();
    }
    default_prog.Unbind();

    return 0;
}
