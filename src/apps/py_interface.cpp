// Modified from cli.cpp
//

#include <span>
#include <iostream>
#include <vector>
#include <string>

#include <cxxopts.hpp>

#include <core/stl/format.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <base/scene.h>
#include <base/pipeline.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/Subdivision.h>
#include <util/thread_pool.h>

#include <backends/ext/denoiser_ext.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"luisa-render-cli"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
    cli.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    cli.add_option("", "D", "define", "Parameter definitions to override scene description macros.",
                   cxxopts::value<std::vector<luisa::string>>()->default_value("<none>"), "<key>=<value>");
    cli.add_option("", "h", "help", "Display this help message", cxxopts::value<bool>()->default_value("false"), "");
    cli.allow_unrecognised_options();
    cli.positional_help("<file>");
    cli.parse_positional("scene");
    auto options = [&] {
        try {
            return cli.parse(argc, argv);
        } catch (const std::exception &e) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to parse command line arguments: {}.",
                e.what());
            std::cout << cli.help() << std::endl;
            exit(-1);
        }
    }();
    if (options["help"].as<bool>()) {
        std::cout << cli.help() << std::endl;
        exit(0);
    }
    if (options["scene"].count() == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("Scene file not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    if (auto unknown = options.unmatched(); !unknown.empty()) [[unlikely]] {
        luisa::string opts{unknown.front()};
        for (auto &&u : luisa::span{unknown}.subspan(1)) {
            opts.append("; ").append(u);
        }
        LUISA_WARNING_WITH_LOCATION(
            "Unrecognized options: {}", opts);
    }
    return options;
}

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

[[nodiscard]] auto parse_cli_macros(int &argc, char *argv[]) {
    SceneParser::MacroMap macros;

    auto parse_macro = [&macros](luisa::string_view d) noexcept {
        if (auto p = d.find('='); p == luisa::string::npos) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid definition: {}", d);
        } else {
            auto key = d.substr(0, p);
            auto value = d.substr(p + 1);
            LUISA_VERBOSE_WITH_LOCATION("Parameter definition: {} = '{}'", key, value);
            if (auto iter = macros.find(key); iter != macros.end()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate definition: {} = '{}'. "
                    "Ignoring the previous one: {} = '{}'.",
                    key, value, key, iter->second);
                iter->second = value;
            } else {
                macros.emplace(key, value);
            }
        }
    };
    // parse all options starting with '-D' or '--define'
    for (int i = 1; i < argc; i++) {
        auto arg = luisa::string_view{argv[i]};
        if (arg == "-D" || arg == "--define") {
            if (i + 1 == argc) {
                LUISA_WARNING_WITH_LOCATION(
                    "Missing definition after {}.", arg);
                // remove the option
                argv[i] = nullptr;
            } else {
                parse_macro(argv[i + 1]);
                // remove the option and its argument
                argv[i] = nullptr;
                argv[++i] = nullptr;
            }
        } else if (arg.starts_with("-D")) {
            parse_macro(arg.substr(2));
            // remove the option
            argv[i] = nullptr;
        }
    }
    // remove all nullptrs
    auto new_end = std::remove(argv, argv + argc, nullptr);
    argc = static_cast<int>(new_end - argv);
    return macros;
}

class _MeshLoader {

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;
    uint _properties{};

public:
    [[nodiscard]] auto mesh() const noexcept { return MeshView{_vertices, _triangles}; }
    [[nodiscard]] auto properties() const noexcept { return _properties; }

    // Load the mesh from a file.
    [[nodiscard]] static auto load(std::filesystem::path path, uint subdiv_level,
                                   bool flip_uv, bool drop_normal, bool drop_uv) noexcept {

        static luisa::lru_cache<uint64_t, std::shared_future<_MeshLoader>> loaded_meshes{256u};
        static std::mutex mutex;

        auto abs_path = std::filesystem::canonical(path).string();
        auto key = luisa::hash_value(abs_path, luisa::hash_value(subdiv_level));


        std::scoped_lock lock{mutex};
        if (auto m = loaded_meshes.at(key)) { return *m; }

        auto future = global_thread_pool().async([path = std::move(path), subdiv_level, flip_uv, drop_normal, drop_uv] {
            Clock clock;
            auto path_string = path.string();
            Assimp::Importer importer;
            importer.SetPropertyInteger(
                AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
            importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 45.f);
            auto import_flags = aiProcess_RemoveComponent | aiProcess_SortByPType |
                                aiProcess_ValidateDataStructure | aiProcess_ImproveCacheLocality |
                                aiProcess_PreTransformVertices | aiProcess_FindInvalidData |
                                aiProcess_JoinIdenticalVertices;
            auto remove_flags = aiComponent_ANIMATIONS | aiComponent_BONEWEIGHTS |
                                aiComponent_CAMERAS | aiComponent_LIGHTS |
                                aiComponent_MATERIALS | aiComponent_TEXTURES |
                                aiComponent_COLORS | aiComponent_TANGENTS_AND_BITANGENTS;
            if (drop_uv) {
                remove_flags |= aiComponent_TEXCOORDS;
            } else {
                if (!flip_uv) { import_flags |= aiProcess_FlipUVs; }
                import_flags |= aiProcess_GenUVCoords | aiProcess_TransformUVCoords;
            }
            if (drop_normal) {
                import_flags |= aiProcess_DropNormals;
                remove_flags |= aiComponent_NORMALS;
            } else {
                import_flags |= aiProcess_GenSmoothNormals;
            }
            if (subdiv_level == 0u) { import_flags |= aiProcess_Triangulate; }
            importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, static_cast<int>(remove_flags));
            auto model = importer.ReadFile(path_string.c_str(), import_flags);
            if (model == nullptr || (model->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
                model->mRootNode == nullptr || model->mRootNode->mNumMeshes == 0) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Failed to load mesh '{}': {}.",
                    path_string, importer.GetErrorString());
            }
            if (auto err = importer.GetErrorString();
                err != nullptr && err[0] != '\0') [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Mesh '{}' has warnings: {}.",
                    path_string, err);
            }
            LUISA_ASSERT(model->mNumMeshes == 1u, "Only single mesh is supported.");
            auto mesh = model->mMeshes[0];
            if (subdiv_level > 0u) {
                auto subdiv = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
                aiMesh *subdiv_mesh = nullptr;
                subdiv->Subdivide(mesh, subdiv_mesh, subdiv_level, true);
                model->mMeshes[0] = nullptr;
                mesh = subdiv_mesh;
                delete subdiv;
            }
            if (mesh->mTextureCoords[0] == nullptr ||
                mesh->mNumUVComponents[0] != 2) [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid texture coordinates in mesh '{}': "
                    "address = {}, components = {}.",
                    path_string,
                    static_cast<void *>(mesh->mTextureCoords[0]),
                    mesh->mNumUVComponents[0]);
            }
            auto vertex_count = mesh->mNumVertices;
            auto ai_positions = mesh->mVertices;
            auto ai_normals = mesh->mNormals;
            auto ai_uvs = mesh->mTextureCoords[0];
            LUISA_INFO("normals: {}", ai_normals ? 1 : 0);
            _MeshLoader loader;
            loader._vertices.resize(vertex_count);
            if (ai_normals) { loader._properties |= Shape::property_flag_has_vertex_normal; }
            if (ai_uvs) { loader._properties |= Shape::property_flag_has_vertex_uv; }
            for (auto i = 0; i < vertex_count; i++) {
                auto p = make_float3(ai_positions[i].x, ai_positions[i].y, ai_positions[i].z);
                auto n = ai_normals ?
                             normalize(make_float3(ai_normals[i].x, ai_normals[i].y, ai_normals[i].z)) :
                             make_float3(0.f, 0.f, 1.f);
                auto uv = ai_uvs ? make_float2(ai_uvs[i].x, ai_uvs[i].y) : make_float2(0.f, 0.f);
                loader._vertices[i] = Vertex::encode(p, n, uv);
            }
            if (subdiv_level == 0u) {
                auto ai_triangles = mesh->mFaces;
                loader._triangles.resize(mesh->mNumFaces);
                for (auto i = 0; i < mesh->mNumFaces; i++) {
                    auto &&face = ai_triangles[i];
                    assert(face.mNumIndices == 3u);
                    loader._triangles[i] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
                }
            } else {
                auto ai_quads = mesh->mFaces;
                loader._triangles.resize(mesh->mNumFaces * 2u);
                for (auto i = 0u; i < mesh->mNumFaces; i++) {
                    auto &&face = ai_quads[i];
                    assert(face.mNumIndices == 4u);
                    loader._triangles[i * 2u + 0u] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
                    loader._triangles[i * 2u + 1u] = {face.mIndices[2], face.mIndices[3], face.mIndices[0]};
                }
            }
            LUISA_INFO("Loaded triangle mesh '{}' in {} ms.", path_string, clock.toc());
            return loader;
        });
        loaded_meshes.emplace(key, future);
        return future;
    }
};

namespace py = pybind11;
using namespace py::literals;
using buffer_t = py::array_t<float>;

DenoiserExt *denoiser_ext = nullptr;
Stream stream;
Buffer<float> hdr_buffer, denoised_buffer;

void init(const std::string &context_path, const std::string &scene_file, uint32_t cuda_device){
    // auto context_path = "/home/winnie/LuisaRender/build/bin";

    log_level_info();
    luisa::compute::Context context{context_path};

    luisa::string backend = "CUDA";
    // std::filesystem::path path("/home/winnie/scene/glass-of-water/scene.luisa");
    std::filesystem::path path(scene_file);

    compute::DeviceConfig config;
    config.device_index = cuda_device;
    auto device = context.create_device(backend, &config);

    SceneParser::MacroMap macros;
    Clock clock;
    auto scene_desc = SceneParser::parse(path, macros);
    auto parse_time = clock.toc();
;
    LUISA_INFO("Parsed scene description file '{}' in {} ms.",
               path.string(), parse_time);

    auto desc = scene_desc.get();
    auto template_node = desc->node("template_1");

    auto scene = Scene::create(context, desc);

    denoiser_ext = device.extension<DenoiserExt>();
    stream = device.create_stream(StreamTag::COMPUTE);

    DenoiserExt::DenoiserMode mode{};

    auto resolution = make_uint2(256u, 256u);
    auto channel_count = 4u;

    hdr_buffer = device.create_buffer<float>(resolution.x * resolution.y * 4);
    denoised_buffer = device.create_buffer<float>(resolution.x * resolution.y * 4);

    DenoiserExt::DenoiserInput data;
    data.beauty = &hdr_buffer;

    denoiser_ext->init(stream, mode, data, resolution);

    // for (auto i = 0u; i < mesh_pool.size(); i++) {
    // for (auto i = 0u; i < 1; i++) {
    // }
}

void add_rigid_body(const std::string &id, ) {

}

void render_frame(py::dict deformable_mesh) {
    Geometry::TemplateMapping mapping;
    for (auto it : deformable_mesh) {
        auto mesh = py::cast<py::tuple>(it.second);
        auto vertices = py::cast<py::array_t<float>>(mesh[0]);
        auto triangles = py::cast<py::array_t<uint32_t>>(mesh[1]);
        auto v = vertices.unchecked<2>();
        auto f = triangles.unchecked<2>();
        LUISA_INFO("{} => vertices: {}, triangles: {}", py::cast<std::string>(it.first), vertices.ndim(), triangles.ndim());
        mapping[it.first] = 
    }
    std::filesystem::path save_path(luisa::format("/home/winnie/LuisaRender/render/{}.exr", i));
    std::filesystem::path save_path_denoised(luisa::format("/home/winnie/LuisaRender/render/{}_denoised.exr", i));
    Geometry::TemplateMapping mapping;
    mapping["liquid"] = mesh_pool[i];
    auto pipeline = Pipeline::create(device, stream, *scene, mapping);
    auto buffer = pipeline->render_to_buffer(stream);
    stream.synchronize();
    save_image(save_path, buffer, scene->cameras()[0]->film()->resolution());
    stream << hdr_buffer.copy_from(buffer);
    stream.synchronize();
    denoiser_ext->process(stream, data);
    denoiser_ext->get_result(stream, denoised_buffer);
    stream.synchronize();
    float *new_buffer = new float[256*256*4];
    stream << denoised_buffer.copy_to(new_buffer);
    stream.synchronize();
    save_image(save_path_denoised, new_buffer, scene->cameras()[0]->film()->resolution());
}

void destroy() {
    if (denoiser_ext != nullptr) {
        denoiser_ext->destroy(stream);
    }
}

PYBIND11_MODULE(LuisaRenderPy, m) {
    m.doc() = "Python binding of LuisaRender";
    m.def("init", &init);
    m.def("destroy", &destroy);
    m.def("add_body", &add_rigid_body);
    m.def("add_deformable_body", &add_deformable_body);
    m.def("render_frame", &render_frame);
}