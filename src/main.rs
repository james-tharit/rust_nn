use eframe::egui;
use rust_nn::{layer::Layer, nueron::Nueron};

struct App {
    inputs: Vec<f32>,
    layer: Layer,
}

impl Default for App {
    fn default() -> Self {
        let n1 = Nueron::new(vec![0.2, 0.8, -0.5, 1.0], 2.0);
        let n2 = Nueron::new(vec![0.5, -0.91, 0.26, -0.5], 3.0);
        let n3 = Nueron::new(vec![-0.26, -0.27, 0.17, 0.87], 0.5);
        Self {
            inputs: vec![1.0, 2.0, 3.0, 2.5],
            layer: Layer::new(vec![n1, n2, n3]),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::SidePanel::left("inputs")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Inputs");
                for (i, v) in self.inputs.iter_mut().enumerate() {
                    ui.add(egui::Slider::new(v, -5.0..=5.0).text(format!("x{i}")));
                }
                ui.separator();
                ui.label("Edges: blue = positive weight, red = negative. Thickness = |weight|.");
                ui.label("Output neuron shade tracks sigmoid activation (0..1).");
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let outputs = self.layer.output(&self.inputs);
            let (resp, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());
            let rect = resp.rect;

            let neurons = self.layer.neurons();
            let n_in = self.inputs.len();
            let n_out = neurons.len();
            let in_x = rect.left() + 70.0;
            let out_x = rect.right() - 90.0;
            let spread = |n: usize, i: usize| -> f32 {
                let pad = 60.0;
                let avail = (rect.height() - 2.0 * pad).max(1.0);
                rect.top() + pad + i as f32 * avail / (n.saturating_sub(1).max(1)) as f32
            };

            // edges first so nodes sit on top
            for (j, neuron) in neurons.iter().enumerate() {
                for (i, w) in neuron.weights.iter().enumerate() {
                    let color = if *w >= 0.0 {
                        egui::Color32::from_rgb(60, 120, 220)
                    } else {
                        egui::Color32::from_rgb(220, 80, 80)
                    };
                    let stroke = egui::Stroke::new((w.abs() * 2.0).clamp(0.5, 6.0), color);
                    painter.line_segment(
                        [
                            egui::pos2(in_x, spread(n_in, i)),
                            egui::pos2(out_x, spread(n_out, j)),
                        ],
                        stroke,
                    );
                    // weight label near the neuron end
                    let t = 0.78;
                    let lx = in_x + (out_x - in_x) * t;
                    let ly = spread(n_in, i) + (spread(n_out, j) - spread(n_in, i)) * t;
                    painter.text(
                        egui::pos2(lx, ly - 8.0),
                        egui::Align2::CENTER_CENTER,
                        format!("{w:.2}"),
                        egui::FontId::proportional(10.0),
                        color,
                    );
                }
            }

            for (i, v) in self.inputs.iter().enumerate() {
                let p = egui::pos2(in_x, spread(n_in, i));
                painter.circle_filled(p, 22.0, egui::Color32::from_gray(210));
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("{v:.2}"),
                    egui::FontId::proportional(12.0),
                    egui::Color32::BLACK,
                );
            }

            for (j, a) in outputs.iter().enumerate() {
                let shade = (a * 255.0).clamp(0.0, 255.0) as u8;
                let p = egui::pos2(out_x, spread(n_out, j));
                painter.circle_filled(
                    p,
                    28.0,
                    egui::Color32::from_rgb(shade, shade, 255),
                );
                let txt_color = if *a > 0.5 {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::BLACK
                };
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("{a:.3}"),
                    egui::FontId::proportional(13.0),
                    txt_color,
                );
                painter.text(
                    p + egui::vec2(0.0, 42.0),
                    egui::Align2::CENTER_CENTER,
                    format!("bias {:.2}", neurons[j].bias),
                    egui::FontId::proportional(11.0),
                    egui::Color32::GRAY,
                );
            }

            ctx.request_repaint();
        });
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "rust_nn viz",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::<App>::default())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_neuron_layer_output() {
        let inputs: Vec<f32> = vec![1.0, 2.0, 3.0, 2.5];
        let n1 = Nueron::new(vec![0.2, 0.8, -0.5, 1.0], 2.0);
        let layer = Layer::new(vec![n1]);

        let output: Vec<f32> = layer.output(&inputs);
        // sigmoid(4.8) ≈ 0.9918
        assert!((output[0] - 0.9918).abs() < 1e-3);
    }
}
