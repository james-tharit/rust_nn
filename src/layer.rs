use crate::nueron::Nueron;

#[derive(Debug)]
pub struct Layer {
    nuerons: Vec<Nueron>,
}
impl Layer {
    pub fn new(nuerons: Vec<Nueron>) -> Layer {
        Layer { nuerons }
    }
    pub fn len(&self) -> usize {
        self.nuerons.len()
    }

    pub fn output(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::new();
        for n in &self.nuerons {
            output.push(n.output(inputs));
        }
        output
    }
}
