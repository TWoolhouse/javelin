use circular_buffer::CircularBuffer;
use cpal::traits::*;
use cpal::Sample;
use derive_more::derive::{Deref, DerefMut};
use hound;
use ringbuf::storage::Heap;
use ringbuf::{traits::*, LocalRb, StaticRb};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Pow;
use std::f32::consts::PI;
use std::num::NonZeroU32;
use std::ops::DerefMut;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use tiny_skia::{Color, Pixmap, Transform};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use winit::window::Window;

const FFT_SIZE: usize = nearest23(8192) as usize;
const AMP: f32 = 0.1;

const fn nearest23(target: i64) -> i64 {
    let mut num = 1;
    while num < target {
        num <<= 1;
    }

    let mut best = num;
    loop {
        if (num - target).abs() < (best - target).abs() {
            best = num;
        }

        if num < target {
            num *= 3;
        } else if num % 2 == 0 {
            num /= 2;
        } else {
            return best;
        }
    }
}

fn fft(samples: &[f32], draw: &mut DrawCtx) -> Vec<Complex<f32>> {
    let size = samples.len();
    // let fft_fwd = draw.fft_planner.plan_fft_forward(size);
    let fft_fwd = draw.fft_planner.plan_fft_forward(size);
    let mut fft_buf_samples: Vec<Complex<f32>> =
        samples.into_iter().map(|x| Complex::new(*x, 0.0)).collect();
    let mut scratch = vec![Complex::new(0.0, 0.0); size];
    fft_fwd.process_with_scratch(&mut fft_buf_samples, &mut scratch);
    fft_buf_samples.truncate(size / 2 + 1);
    fft_buf_samples
}
struct RenderTarget {
    // SAFETY: surface must be dropped before window
    surface: softbuffer::Surface<DisplayHandle<'static>, WindowHandle<'static>>,
    window: Pin<Box<Window>>,
    pixmap: Pixmap,
}

impl RenderTarget {
    fn new(window: Window, context: &softbuffer::Context<DisplayHandle<'static>>) -> Self {
        let window = Box::pin(window);
        let mut surface = softbuffer::Surface::new(context, unsafe {
            std::mem::transmute(window.as_ref().window_handle().unwrap())
        })
        .unwrap();
        let size = window.inner_size();
        surface
            .resize(
                NonZeroU32::new(size.width).unwrap(),
                NonZeroU32::new(size.height).unwrap(),
            )
            .unwrap();
        RenderTarget {
            window,
            surface,
            pixmap: Pixmap::new(size.width, size.height).unwrap(),
        }
    }
}

fn pixmap_write_to_buffer(
    pixmap: &Pixmap,
    buffer: &mut softbuffer::Buffer<'_, DisplayHandle<'static>, WindowHandle<'static>>,
) {
    let buf_dst = buffer.as_mut();
    let buf_src = pixmap.data();

    buf_dst
        .iter_mut()
        .zip(buf_src.chunks_exact(4).map(|pixel| {
            // RGBA -> 0RGB
            (pixel[0] as u32) << 16 | (pixel[1] as u32) << 8 | (pixel[2] as u32) << 0
        }))
        .for_each(|(dst, src)| {
            *dst = src;
        });
}

fn draw_line_graph(pixmap: &mut Pixmap, data: &[f32], y_max: f32) {
    let width = pixmap.width() as usize;
    let height = pixmap.height() as usize;

    let x_sf = width as f32 / data.len() as f32;
    let y_sf = height as f32 / y_max;

    let mut pb = tiny_skia::PathBuilder::new();
    // pb.move_to(0.0, 0.0);

    for (x, dp) in data.iter().enumerate() {
        let y = (dp * y_sf) as usize;
        let x = x as f32 * x_sf;
        pb.move_to(x, 0.0);
        pb.line_to(x, y as f32);
    }
    // pb.line_to(width as f32, 0.0);

    let mut paint = tiny_skia::Paint::default();
    paint.set_color_rgba8(10, 67, 227, 255);
    paint.anti_alias = true;

    let path = pb.finish().unwrap();
    let mut stroke = tiny_skia::Stroke::default();
    stroke.width = 5.0;
    stroke.line_cap = tiny_skia::LineCap::Round;

    pixmap.stroke_path(
        &path,
        &paint,
        &stroke,
        Transform::identity()
            .post_scale(1.0, -1.0)
            .post_translate(0.0, height as f32),
        None,
    );
}

fn frequency_to_note(frequency: f32) -> f32 {
    if frequency < 8.2 {
        return 0.0;
    }
    12.0 * (frequency / 440.0).log2() + 69.0
}

fn hann_window(samples: &mut [f32]) {
    let size = samples.len();
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / size as f32).cos();
        *s *= w;
    }
}

fn freq_mags(fft: &[Complex<f32>]) -> Vec<f32> {
    fft.iter()
        .map(|c| {
            let im = c.im.abs();
            let re = c.re.abs();
            im.max(re)
        })
        .collect()
}

fn bucket_note(freq_mags: &[f32]) -> Vec<f32> {
    let mut notes = vec![0.0f32; frequency_to_note(freq_mags.len() as f32) as usize + 1];
    for (i, mag) in freq_mags.iter().enumerate() {
        let note = frequency_to_note(i as f32);
        notes[note as usize] += mag;
    }
    notes.remove(0);
    notes
}

fn bucket_tsoding(freq_mags: &[f32]) -> Vec<f32> {
    let mult = 1.06f32;
    let freq_start = 20.0f32;
    let freq_end = freq_mags.len() as f32;
    let mut freq_lb = freq_start;
    let mut out: Vec<f32> = Vec::with_capacity(100);

    loop {
        let freq_ub = freq_lb * mult;
        let mut amp: f32 = freq_mags[(freq_lb as usize)..(freq_ub as usize).min(freq_end as usize)]
            .iter()
            .sum();
        amp /= (freq_ub - freq_lb) + 1.0;
        out.push(amp);

        freq_lb = freq_ub;
        if freq_lb > freq_end {
            break;
        }
    }
    out
}

fn bucket_e(freq_mags: &[f32]) -> Vec<f32> {
    freq_mags.iter().map(|m| m.exp() - 1.0).collect()
}

fn main_draw(pixmap: &mut Pixmap, draw: &mut DrawCtx) {
    let mut samples = {
        let samples = draw.samples.lock().unwrap();
        let size = samples.occupied_len();
        if size < FFT_SIZE {
            return;
        }
        samples.iter().cloned().collect::<Vec<f32>>()
    };
    hann_window(&mut samples);
    let fft_res = fft(&samples, draw);
    let freq_mags = freq_mags(&fft_res);

    let mut output = bucket_tsoding(&freq_mags);

    let mag_max = {
        static mut MAG_MAX_HISTORY: CircularBuffer<100, f32> = CircularBuffer::new();
        let mag_max = output
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .to_owned();
        unsafe {
            MAG_MAX_HISTORY.push_back(mag_max);
            MAG_MAX_HISTORY
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .to_owned()
                .max(0.5)
        }
    };

    let mag_max = if false {
        output.iter_mut().for_each(|m| {
            *m = 1.0f32 - (1.0 - (*m / mag_max)).pow(8);
        });
        1.0
    } else {
        mag_max
    };

    pixmap.fill(Color::BLACK);
    draw_line_graph(pixmap, &output, mag_max);
}

struct DrawCtx {
    samples: Arc<Mutex<LocalRb<Heap<f32>>>>,
    playing: Arc<Mutex<bool>>,
    fft_planner: rustfft::FftPlanner<f32>,
}

impl DrawCtx {
    fn new() -> Self {
        DrawCtx {
            samples: Arc::new(Mutex::new(LocalRb::new(FFT_SIZE))),
            fft_planner: rustfft::FftPlanner::<f32>::new(),
            playing: Arc::new(Mutex::new(false)),
        }
    }
}

#[derive(Deref, DerefMut)]
struct Application {
    #[deref]
    #[deref_mut]
    target: Option<Mutex<RenderTarget>>,
    context: softbuffer::Context<DisplayHandle<'static>>,
    draw: DrawCtx,
}

impl Application {
    fn new<T>(event_loop: &EventLoop<T>, ctx: DrawCtx) -> Self {
        Application {
            target: None,
            context: softbuffer::Context::new(unsafe {
                std::mem::transmute(event_loop.display_handle().unwrap())
            })
            .unwrap(),
            draw: ctx,
        }
    }
    fn target(&mut self) -> MutexGuard<RenderTarget> {
        self.target.as_mut().unwrap().lock().unwrap()
    }
}

#[derive(Debug)]
enum AppEvent {
    Redraw,
}

impl ApplicationHandler<AppEvent> for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attrs = winit::window::WindowAttributes::default()
            .with_title("Test")
            .with_active(true)
            .with_inner_size(LogicalSize::new(800.0, 600.0));
        let window = event_loop.create_window(window_attrs).unwrap();
        let target = RenderTarget::new(window, &self.context);
        self.target = Some(Mutex::new(target));
    }
    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.target = None;
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::Resized(size) => {
                println!("{:?}", size);
                let mut target = self.target();
                target
                    .surface
                    .resize(
                        NonZeroU32::new(size.width).unwrap(),
                        NonZeroU32::new(size.height).unwrap(),
                    )
                    .unwrap();
                target.pixmap = Pixmap::new(size.width, size.height).unwrap();
            }
            winit::event::WindowEvent::RedrawRequested => {
                let mut target = self.target.as_mut().unwrap().lock().unwrap();
                let target = target.deref_mut();
                main_draw(&mut target.pixmap, &mut self.draw);

                let mut buf = target.surface.buffer_mut().unwrap();
                pixmap_write_to_buffer(&target.pixmap, &mut buf);
                target.window.pre_present_notify();
                buf.present().unwrap();
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        Key::Named(NamedKey::Space) => {
                            let mut playing = self.draw.playing.lock().unwrap();
                            *playing = !*playing;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: AppEvent) {
        if self.target.is_some() {
            match event {
                AppEvent::Redraw => {
                    self.target().window.request_redraw();
                }
            }
        }
    }
}

fn main() {
    println!("FFT_SIZE = {}", FFT_SIZE);
    let event_loop = EventLoop::<AppEvent>::with_user_event().build().unwrap();
    let event_loop_proxy = event_loop.create_proxy();

    let draw = DrawCtx::new();
    let recent_samples = Arc::clone(&draw.samples);
    let playing = Arc::clone(&draw.playing);

    {
        // `app` must drop before `event_loop``
        let mut app = Application::new(&event_loop, draw);

        let host = cpal::default_host();
        let device = host.default_output_device().unwrap();
        let config = device.default_output_config().unwrap();
        dbg!(&config);

        let mic_dev = host.default_input_device().unwrap();
        let mic_config = mic_dev.default_input_config().unwrap();
        dbg!(&mic_config);

        let mut reader = hound::WavReader::open("asset/48000.wav").unwrap();
        let spec = reader.spec();
        println!("{:?}", &spec);
        let vec_samples = reader
            .samples::<i16>()
            .map(|x| x.unwrap().to_sample::<f32>())
            .collect::<Vec<_>>();
        let vec_samples: &'static Vec<f32> = unsafe { std::mem::transmute(&vec_samples) };
        let mut it_samples = vec_samples.iter().cycle();

        let (mut mic_tx, mut mic_rx) = {
            let mic_buf: StaticRb<f32, 1024> = Default::default();
            mic_buf.split()
        };

        let mic = mic_dev
            .build_input_stream(
                &mic_config.clone().into(),
                move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                    mic_tx.push_iter(data.iter().step_by(mic_config.channels() as usize).cloned());
                },
                move |err| {
                    eprintln!("an error occurred on stream: {}", err);
                },
                None,
            )
            .unwrap();

        let stream = device
            .build_output_stream(
                &config.clone().into(),
                move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    let playing = {
                        let playing = playing.lock().unwrap();
                        *playing
                    };

                    if playing {
                        for (channels, s) in data.chunks_exact_mut(2).zip(it_samples.by_ref()) {
                            for chl in channels.iter_mut() {
                                *chl = s * AMP;
                            }
                        }
                    } else {
                        // for chl in data.iter_mut() {
                        //     *chl = 0.0;
                        // }
                        let channels = config.channels() as usize;
                        for (channels, s) in data.chunks_exact_mut(channels).zip(mic_rx.pop_iter())
                        {
                            for chl in channels.iter_mut() {
                                *chl = s * AMP;
                            }
                        }
                    }

                    let redraw = {
                        let mut samples = recent_samples.lock().unwrap();
                        samples.push_iter_overwrite(data.iter().step_by(2).cloned());
                        // samples.len() >= FFT_SIZE
                        true
                    };
                    if redraw {
                        event_loop_proxy.send_event(AppEvent::Redraw).unwrap();
                    }
                },
                move |err| {
                    eprintln!("an error occurred on stream: {}", err);
                },
                None,
            )
            .unwrap();
        stream.play().unwrap();
        mic.play().unwrap();

        event_loop.run_app(&mut app).unwrap();
    }

    // const FILENAME: &str = "asset/am.wav";
    // now_i_know_what_im_doing(FILENAME, 1024, 1);
}
