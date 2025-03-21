#include <luisa/core/stl/format.h>
#include <luisa/core/basic_types.h>
#include <luisa/backends/ext/denoiser_ext.h>
#include <apps/py_class.h>
#include <base/scene.h>
#include <base/pipeline.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

class PyScene {
public:
    Device &_device;
    Context &_context;
    Stream &_stream;

    luisa::unique_ptr<Pipeline> _pipeline{nullptr};
    luisa::unique_ptr<Scene> _scene{nullptr};
    luisa::unique_ptr<SceneDesc> _scene_desc{nullptr};
    DenoiserExt *denoiser_ext{nullptr};

    luisa::vector<SceneNodeDesc *> shapes;
    luisa::vector<SceneNodeDesc *> cameras;

    PyScene(Device &device, Context &context, Stream &stream) noexcept:
        _device{device}, _context{context}, _stream{stream} { }
    
    void init(PyRender *render) noexcept {
        /* build scene and pipeline */
        Clock clock;
        _scene_desc = luisa::make_unique<SceneDesc>();
        auto scene_node = _scene_desc.get();
        render->define_in_scene(scene_node);

        _scene = Scene::create(_context, scene_node);
        auto scene_create_time = clock.toc();
        LUISA_INFO("Scene created in {} ms.", scene_create_time);
        LUISA_INFO("Create {}: {}", render->node()->identifier(), _scene->info());

        _pipeline = Pipeline::create(_device, *_scene);
        auto pipeline_create_time = clock.toc();
        LUISA_INFO("Pipeline created in {} ms.", pipeline_create_time - scene_create_time);

        denoiser_ext = _device.extension<DenoiserExt>();        // build denoiser
    }

    void update_environment(PyEnvironment *environment) noexcept {
        environment->define_in_scene(_scene_desc.get());
        auto environment_node = _scene->update_environment(environment->node());
        LUISA_INFO("Update {}: {}", environment->node()->identifier(), environment_node->info());
    }

    void update_emission(PyLight *light) noexcept {
        light->define_in_scene(_scene_desc.get());
    }

    void update_subsurface(PySubsurface *subsurface) noexcept {
        subsurface->define_in_scene(_scene_desc.get());
    }

    void update_surface(PySurface *surface) noexcept {
        surface->define_in_scene(_scene_desc.get());
    }

    void update_shape(PyShape *shape) noexcept {
        shape->define_in_scene(_scene_desc.get());
        auto shape_node = _scene->update_shape(shape->node());
        LUISA_INFO("Update {}: {}", shape->node()->identifier(), shape_node->info());
    }

    void update_camera(PyCamera *camera, bool denoise) noexcept {
        camera->define_in_scene(_scene_desc.get());
        auto camera_node = _scene->update_camera(camera->node());
        LUISA_INFO("Update {}: {}", camera->node()->identifier(), camera_node->info());
        
        if (camera->camera == nullptr) {
            const auto &resolution = camera_node->film()->resolution();
            uint pixel_count = resolution.x * resolution.y;
            camera->camera = (void *)camera_node;
            camera->denoise = denoise;
            if (denoise) {
                if (denoiser_ext == nullptr)
                    LUISA_ERROR_WITH_LOCATION("Denoiser is not supported on this platform.");
                camera->color_buffer = luisa::make_unique<Buffer<float4>>(
                    _device.create_buffer<float4>(pixel_count));
                camera->denoised_buffer = luisa::make_unique<Buffer<float4>>(
                    _device.create_buffer<float4>(pixel_count));
                camera->denoiser = denoiser_ext->create(_stream);
                auto input = DenoiserExt::DenoiserInput(resolution.x, resolution.y);
                input.push_noisy_image(
                    camera->color_buffer->view(),
                    camera->denoised_buffer->view(),
                    DenoiserExt::ImageFormat::FLOAT3,
                    DenoiserExt::ImageColorSpace::HDR
                );
                input.noisy_features = false;
                input.filter_quality = DenoiserExt::FilterQuality::DEFAULT;
                input.prefilter_mode = DenoiserExt::PrefilterMode::NONE;
                camera->denoiser->init(input);
            }
        }
    }

    void update_scene(float time) noexcept {
        _pipeline->set_time(time);
        _pipeline->update(_stream);
    }

    py::bytes render_frame(PyCamera *camera) noexcept {
        auto camera_node = static_cast<Camera *>(camera->camera);
        const auto &resolution = camera_node->film()->resolution();
        luisa::vector<float4> buffer;
        _pipeline->render_to_buffer(_stream, camera_node, buffer);

        _stream.synchronize();
        auto buffer_p = reinterpret_cast<float *>(buffer.data());

        if (camera->denoise) {  // denoise image 
            Clock clock;
            _stream << camera->color_buffer->copy_from(buffer_p) << synchronize();
            camera->denoiser->execute(true);
            _stream << camera->denoised_buffer->copy_to(buffer_p) << synchronize();
            auto denoise_time = clock.toc();
            LUISA_INFO("Denoised image in {} ms", denoise_time);
        }

        apply_gamma(buffer_p, resolution);

        luisa::vector<uint8_t> byte_buffer;
        byte_buffer.resize(resolution.x * resolution.y * 4);
        auto byte_buffer_p = reinterpret_cast<uint8_t *>(byte_buffer.data());
        convert_uint8(byte_buffer_p, buffer_p, resolution);

        return py::bytes(reinterpret_cast<const char *>(byte_buffer.data()), byte_buffer.size());
    }
};