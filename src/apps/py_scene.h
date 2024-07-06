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
    // std::mutex _stream_mutex;

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

        // std::scoped_lock lock{_stream_mutex};
        _pipeline = Pipeline::create(_device, _stream, *_scene);
        auto pipeline_create_time = clock.toc();
        LUISA_INFO("Pipeline created in {} ms.", pipeline_create_time - scene_create_time);

        denoiser_ext = _device.extension<DenoiserExt>();        // build denoiser
    }

    void update_environment(PyEnvironment *environment) noexcept {
        environment->define_in_scene(_scene_desc.get());
        auto environment_node = _scene->update_environment(environment->node());
        // _scene_desc->root()->add_property("environment", environment->node());
        // auto environment_node = 
        // LUISA_INFO("Add: {}", environment_info.get_info());
        // auto environment_node = scene->add_environment(environment_info);
    }

    void update_emission(PyLight *light) noexcept {
        light->define_in_scene(_scene_desc.get());
        // auto light_node = _scene->load_light(light->node());
        // LUISA_INFO("Add: {}", light_info.get_info());
        // auto light_node = scene->add_light(light_info);
    }

    void update_surface(PySurface *surface) noexcept {
        surface->define_in_scene(_scene_desc.get());
        // auto surface_node = _scene->load_surface(surface->node());
        // LUISA_INFO("Add: {}", surface.surface_info.get_info());
        // auto surface_node = scene->add_surface(surface.surface_info);
    }

    void update_shape(PyShape *shape) noexcept {
        shape->define_in_scene(_scene_desc.get());
        auto shape_node = _scene->update_shape(shape->node(), !shape->loaded);
        if (!shape->loaded) shape->loaded = true;
        // shapes.emplace_back(shape->node());
        // auto shape_node = _scene->load_shape(shape->node());
        // LUISA_INFO("Update: {}", shape.shape_info.get_info());
        // auto shape_node = scene->update_shape(shape.shape_info);
    }

    void update_camera(PyCamera *camera, bool denoise) noexcept {
        camera->define_in_scene(_scene_desc.get());
        auto [camera_node, camera_index] = _scene->update_camera(camera->node(), !camera->loaded);
        
        if (!camera->loaded) {
            const auto &resolution = camera_node->film()->resolution();
            uint pixel_count = resolution.x * resolution.y;
            camera->loaded = true;
            camera->index = camera_index;
            camera->denoise = denoise;
            if (denoise) {
                camera->color_buffer = luisa::make_unique<Buffer<float4>>(
                    _device.create_buffer<float4>(pixel_count));
                camera->denoised_buffer = luisa::make_unique<Buffer<float4>>(
                    _device.create_buffer<float4>(pixel_count)); 
            }
        }
    }

    PyDoubleArr render_frame(PyCamera *camera, float time) noexcept {
        // _scene->update(_scene_desc.get());
        _pipeline->scene_update(_stream, *_scene, time);

        auto idx = camera->index;
        const auto &resolution = _scene->cameras()[idx]->film()->resolution();

        luisa::vector<float4> buffer;
        _pipeline->render_to_buffer(_stream, idx, buffer);
        _stream.synchronize();
        auto buffer_p = reinterpret_cast<float *>(buffer.data());

        if (camera->denoise) {  // denoise image 
            Clock clock;
            auto color_buffer = camera->color_buffer.get();
            auto denoised_buffer = camera->denoised_buffer.get();
            auto denoiser = denoiser_ext->create(_stream);
            {
                auto input = DenoiserExt::DenoiserInput{resolution.x, resolution.y};
                input.push_noisy_image(
                    color_buffer->view(), denoised_buffer->view(),
                    DenoiserExt::ImageFormat::FLOAT3,
                    DenoiserExt::ImageColorSpace::HDR
                );
                input.noisy_features = false;
                input.filter_quality = DenoiserExt::FilterQuality::DEFAULT;
                input.prefilter_mode = DenoiserExt::PrefilterMode::NONE;
                denoiser->init(input);
            }
            _stream << color_buffer->copy_from(buffer_p) << synchronize();
            denoiser->execute(true);
            _stream << denoised_buffer->copy_to(buffer_p) << synchronize();
            auto denoise_time = clock.toc();
            LUISA_INFO("Denoised image in {} ms", denoise_time);
        }

        apply_gamma(buffer_p, resolution);
        auto array_buffer = PyDoubleArr(resolution.x * resolution.y * 4);
        std::memcpy(array_buffer.mutable_data(), buffer_p, array_buffer.size() * sizeof(float));
        return array_buffer;
    }
};