use std::{
    collections::VecDeque,
    f32::consts::PI,
    sync::{atomic::AtomicUsize, Arc, Mutex},
    time::Duration,
};

use cpal::{traits::*, Sample};
use egui::ViewportBuilder;
use hertz::FFTHertz;
use ringbuf::{traits::*, LocalRb};
use rodio::Source;
use rustfft::num_complex::Complex;

mod gui;
mod hertz;
mod pipeline;
mod playback;

struct AudioCtrl {
    playing: bool,
    input: VecDeque<f32>,
    output: LocalRb<ringbuf::storage::Heap<f32>>,
    mic: Option<LocalRb<ringbuf::storage::Heap<f32>>>,
    cycles: AtomicUsize,
}

impl AudioCtrl {
    fn new() -> Self {
        Self {
            playing: false,
            input: VecDeque::new(),
            output: LocalRb::new(8192),
            mic: Some(LocalRb::new(1024)),
            cycles: AtomicUsize::new(0),
        }
    }
}

struct AudioOutput {
    ctrl: Arc<Mutex<AudioCtrl>>,
    stream: cpal::Stream,
    cfg: cpal::StreamConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderWaveformGraph {
    Raw,
    Semitone,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderWaveformWindow {
    Raw,
    Hann,
}

struct RenderCtrl {
    waveform: Vec<f32>,
    fft_planner: rustfft::FftPlanner<f32>,
    graph_mode: RenderWaveformGraph,
    window: RenderWaveformWindow,
    max_history: VecDeque<(std::time::Instant, f32)>,
    mag_time: f32,
}

impl RenderCtrl {
    fn new() -> Self {
        Self {
            waveform: Vec::new(),
            fft_planner: rustfft::FftPlanner::new(),
            graph_mode: RenderWaveformGraph::Raw,
            window: RenderWaveformWindow::Raw,
            max_history: VecDeque::with_capacity(1024),
            mag_time: 1.0,
        }
    }

    fn fft(&mut self, data: &[f32]) -> Vec<f32> {
        let mut input = data
            .iter()
            .map(|s| Complex::new(*s, 0.0))
            .collect::<Vec<_>>();

        let fft = self.fft_planner.plan_fft_forward(input.len());
        fft.process(&mut input);

        let len_out = input.len() / 2 + 1;
        let output = &mut input[..len_out];
        output.iter().map(|c| c.norm()).collect::<Vec<_>>()
    }

    fn compute_waveform(&mut self, data: &mut [f32]) {
        match self.window {
            RenderWaveformWindow::Raw => {}
            RenderWaveformWindow::Hann => hann_window(data),
        }

        let output = self.fft(data);

        let mut output = match self.graph_mode {
            RenderWaveformGraph::Raw => output,
            RenderWaveformGraph::Semitone => bucket_semitone(&output),
        };

        let time = std::time::Instant::now();
        let max = it_max_f32(output.iter().copied());

        let pp = self.max_history.partition_point(|(t, _)| {
            (time - *t) > std::time::Duration::from_secs_f32(self.mag_time)
        });
        self.max_history.drain(..pp);
        self.max_history.push_back((time, max));

        let max = it_max_f32(self.max_history.iter().map(|(_, m)| *m));

        // TODO: Use rect transform for the normalisation in one step.
        output.iter_mut().for_each(|x| *x /= max);

        self.waveform.clear();
        self.waveform.extend(&output);
    }
}

fn hann_window(samples: &mut [f32]) {
    let size = samples.len();
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / size as f32).cos();
        *s *= w;
    }
}

fn bucket_semitone(data: &[f32]) -> Vec<f32> {
    let freq_start = 20.0f32;
    let freq_end = data.len() as f32;
    let freq_step = (1.0 / 12.0f32).exp2();
    let mut buckets: Vec<f32> = Vec::with_capacity((freq_end / freq_step) as usize);

    let mut freq_lb = freq_start;
    loop {
        let freq_ub = freq_lb * freq_step;
        let freq_ub = freq_ub.min(freq_end);

        let mut mag = 0.0;
        let range = (freq_lb as usize)..(freq_ub as usize);
        for i in range.clone() {
            mag += data[i];
        }
        mag /= range.len() as f32;
        buckets.push(mag);

        freq_lb = freq_ub;
        if freq_lb >= freq_end {
            break;
        }
    }

    buckets
}

struct App {
    render: RenderCtrl,
    output: AudioOutput,
    _mic: cpal::Stream,
    controls: bool,
    max_hertz: FFTHertz,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> App {
        let ctx = cc.egui_ctx.clone();
        let stream_out = spawn_stream_output(move || {
            ctx.request_repaint();
        });
        let stream_in = spawn_stream_input(&stream_out.ctrl);
        let fft_size = stream_out.ctrl.lock().unwrap().output.capacity().get();

        let app = App {
            render: RenderCtrl::new(),
            output: stream_out,
            _mic: stream_in,
            controls: true,
            max_hertz: FFTHertz::from_fft_size(fft_size),
        };

        app
    }
}

fn it_max_f32(it: impl Iterator<Item = f32>) -> f32 {
    it.reduce(f32::max).unwrap()
}

fn paint_waveform(ui: &mut egui::Ui, space: egui::Rect, data: &[f32]) {
    let painter = ui.painter();
    let stroke = egui::Stroke::new(4.0, egui::Color32::LIGHT_BLUE);

    let remap = |x: f32, y: f32| {
        let x = x * (space.max.x - space.min.x) / data.len() as f32 + space.min.x;
        let diff = (space.max.y - space.min.y) / 2.0 * y;
        let y = space.min.y + (space.max.y - space.min.y) / 2.0;
        (x, y + diff, y - diff)
    };

    for (i, dp) in data.iter().enumerate() {
        let (x, y_lb, y_ub) = remap(i as f32, *dp);
        painter.vline(x, egui::Rangef::new(y_lb, y_ub), stroke);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // TODO: This is way too much locking
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::C)) {
            self.controls = !self.controls;
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::M)) {
            let mut ctrl = self.output.ctrl.lock().unwrap();
            ctrl.mic = if ctrl.mic.is_some() {
                None
            } else {
                Some(LocalRb::new(1024))
            };
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space)) {
            let mut ctrl = self.output.ctrl.lock().unwrap();
            ctrl.playing = !ctrl.playing;
            if ctrl.playing {
                self.output.stream.play().unwrap();
            } else {
                self.output.stream.pause().unwrap();
            }
        }

        egui::Window::new("Controls")
            .open(&mut self.controls)
            .show(ctx, |ui| {
                let mut ctrl = self.output.ctrl.lock().unwrap();

                ui.checkbox(&mut ctrl.playing, "Playing");

                ui.label(format!(
                    "Stream Length: {:.2} s",
                    ctrl.input.len() as f32 / self.output.cfg.sample_rate.0 as f32
                ));

                let cpu_usage = frame.info().cpu_usage;
                ui.label(if let Some(usage) = cpu_usage {
                    format!("UI Time: {:.2} ms FPS {:.2}", usage * 1000.0, 1.0 / usage)
                } else {
                    "UI Time: N/A FPS: N/A".to_string()
                });

                let mut mic = ctrl.mic.is_some();
                if ui.checkbox(&mut mic, "Mic").changed() {
                    ctrl.mic = if mic { Some(LocalRb::new(1024)) } else { None };
                }

                egui::ComboBox::from_label("Graph Mode")
                    .selected_text(format!("{:?}", self.render.graph_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.render.graph_mode,
                            RenderWaveformGraph::Raw,
                            "Raw",
                        );
                        ui.selectable_value(
                            &mut self.render.graph_mode,
                            RenderWaveformGraph::Semitone,
                            "Semitone",
                        );
                    });

                egui::ComboBox::from_label("Window")
                    .selected_text(format!("{:?}", self.render.window))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.render.window,
                            RenderWaveformWindow::Raw,
                            "Raw",
                        );
                        ui.selectable_value(
                            &mut self.render.window,
                            RenderWaveformWindow::Hann,
                            "Hann",
                        );
                    });

                if ui.button("Serialise Wave Form Data").clicked() {
                    use std::io::Write;
                    let mut file = std::fs::File::create("waveform.csv").unwrap();
                    writeln!(file, "Magnitude").unwrap();
                    for point in &self.render.waveform {
                        writeln!(file, "{point}").unwrap();
                    }
                }

                let mut max_hertz = self.max_hertz.hertz();
                if ui
                    .add(
                        egui::Slider::new(
                            &mut max_hertz,
                            64..=(FFTHertz::from_hertz(20_000).optimal().hertz()),
                        )
                        .logarithmic(true)
                        .text("Hertz"),
                    )
                    .changed()
                {
                    self.max_hertz = FFTHertz::from_hertz(max_hertz).optimal();
                    if self.max_hertz.fft_size() != ctrl.output.capacity().get() {
                        let mut buffer = LocalRb::new(self.max_hertz.fft_size());
                        buffer.push_iter_overwrite(ctrl.output.pop_iter());
                        ctrl.output = buffer;
                    }
                }

                let fft_size = ctrl.output.capacity().get();
                ui.label(format!(
                    "FFT: {} frames {:.2} ms",
                    fft_size,
                    fft_size as f32 / self.output.cfg.sample_rate.0 as f32 * 1_000.0
                ));

                ui.label(format!(
                    "Cycles: {}",
                    ctrl.cycles.load(std::sync::atomic::Ordering::Acquire)
                ));

                ui.label("Mag History Window");
                ui.add(egui::Slider::new(&mut self.render.mag_time, 0.0..=10.0).text("s"));

                if ui.button("Load WAV").clicked() {
                    let ctrl = Arc::clone(&self.output.ctrl);
                    // TODO: Lot of overhead :(
                    std::thread::spawn(move || {
                        if let Some(filepath) = rfd::FileDialog::new()
                            .add_filter("WAV", &["wav"])
                            .pick_file()
                        {
                            let mut reader = hound::WavReader::open(filepath).unwrap();
                            let spec = reader.spec();
                            println!("{:?}", &spec);
                            // Collect to prevent lock contention
                            let data = reader
                                .samples::<i16>()
                                .step_by(spec.channels as usize)
                                .map(|x| x.unwrap().to_sample::<f32>())
                                .collect::<Vec<_>>();
                            let mut ctrl = ctrl.lock().unwrap();
                            ctrl.input.clear();
                            ctrl.input.extend(data);
                        }
                    });
                }
            });
        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: egui::Color32::BLACK,
                inner_margin: egui::Margin::same(5.0),
                ..Default::default()
            })
            .show(ctx, |ui| {
                // TODO: Get space and rect better? Use rect transform
                let space = ui.available_rect_before_wrap();

                // TODO: Not here
                {
                    {
                        let mut input = {
                            let ctrl = self.output.ctrl.lock().unwrap();
                            ctrl.cycles.store(0, std::sync::atomic::Ordering::Release);
                            ctrl.output.iter().copied().collect::<Vec<_>>()
                        };
                        self.render.compute_waveform(&mut input);
                    }
                }
                paint_waveform(ui, space, &self.render.waveform);
            });
    }
}

fn audio_stream_output(
    ctrl: &mut AudioCtrl,
    config: &cpal::StreamConfig,
    data: &mut [f32],
) -> bool {
    if !ctrl.playing {
        for s in data.iter_mut() {
            *s = 0.0;
        }
        return false;
    }

    fn write<T: Copy + Default>(
        d: impl Iterator<Item = T>,
        data: &mut [T],
        output: &mut impl ringbuf::traits::RingBuffer<Item = T>,
        channels: usize,
    ) {
        for (channels, sample) in data
            .chunks_exact_mut(channels)
            .zip(d.chain(std::iter::repeat(Default::default())))
        {
            for channel in channels {
                *channel = sample;
            }
            output.push_overwrite(sample);
        }
    }

    let frames = data.len() / 2;

    if let Some(mic) = &mut ctrl.mic {
        write(
            mic.pop_iter().take(frames),
            data,
            &mut ctrl.output,
            config.channels as usize,
        );
    } else {
        let avaliable = ctrl.input.len().min(frames);
        write(
            ctrl.input.drain(..avaliable),
            data,
            &mut ctrl.output,
            config.channels as usize,
        );
    };

    ctrl.cycles
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    true
}

fn audio_stream_input(ctrl: &mut AudioCtrl, config: &cpal::StreamConfig, data: &[f32]) {
    if let Some(output) = &mut ctrl.mic {
        for sample in data.chunks_exact(config.channels as usize) {
            output.push_overwrite(sample[0]);
        }
    }
}

fn spawn_stream_output(notify: impl Fn() + std::marker::Send + 'static) -> AudioOutput {
    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();
    let config = device.default_output_config().unwrap();
    let config: cpal::StreamConfig = config.into();

    let ctrl = Arc::new(Mutex::new(AudioCtrl::new()));
    let ctrl_export = Arc::clone(&ctrl);
    let cfg = config.clone();

    let stream = device
        .build_output_stream(
            &config.clone(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if audio_stream_output(&mut ctrl.lock().unwrap(), &config, data) {
                    notify();
                }
            },
            move |err| eprintln!("an error occurred on stream: {err}"),
            None,
        )
        .unwrap();
    stream.play().unwrap();
    ctrl_export.lock().unwrap().playing = true;

    AudioOutput {
        ctrl: ctrl_export,
        cfg,
        stream,
    }
}

fn spawn_stream_input(ctrl: &Arc<Mutex<AudioCtrl>>) -> cpal::Stream {
    let host = cpal::default_host();
    let device = host.default_input_device().unwrap();
    let config = device.default_input_config().unwrap();
    let config: cpal::StreamConfig = config.into();

    let ctrl = Arc::clone(ctrl);
    let stream = device
        .build_input_stream(
            &config.clone(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                audio_stream_input(&mut ctrl.lock().unwrap(), &config, data);
            },
            move |err| eprintln!("an error occurred on stream: {err}"),
            None,
        )
        .unwrap();
    stream.play().unwrap();

    stream
}

fn main() -> eframe::Result {
    eframe::run_native(
        "tw_rave",
        eframe::NativeOptions {
            viewport: ViewportBuilder::default()
                .with_active(true)
                .with_icon(Arc::new(egui::IconData::default()))
                .with_title("My Cool App")
                .with_close_button(true),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
