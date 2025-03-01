interface ContinuousParameters {
	activation?: string;
}

const defaultActivation = 'return -1./pow(2., (0.6*pow(x, 2.)))+1.;';

function continuous(parameters?: ContinuousParameters) {
  const activation = parameters?.activation || defaultActivation;

	const simulation = /*wgsl*/ `
@group(0) @binding(0) var<storage> size: u32;
@group(0) @binding(1) var<storage> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> next_state: array<f32>;
@group(0) @binding(3) var<storage> kernel: array<f32>;

@workgroup_size(8, 8)
@compute 
fn compute_main(@builtin(global_invocation_id) cell: vec3u) {
  let i = index(cell.xy);

  // Perform Convolution of the filter. We will use step function as activation function for now
  // k1 | k2 | k3
  // k4 | k5 | k6
  // k7 | k8 | k9
  var sum = 0.0;
  sum += value(cell.x-1, cell.y-1) * kernel[0];
  sum += value(cell.x, cell.y-1) * kernel[1];
  sum += value(cell.x+1, cell.y-1) * kernel[2];
  sum += value(cell.x-1, cell.y) * kernel[3];
  sum += value(cell.x, cell.y) * kernel[4];
  sum += value(cell.x+1, cell.y) * kernel[5];
  sum += value(cell.x-1, cell.y+1) * kernel[6];
  sum += value(cell.x, cell.y+1) * kernel[7];
  sum += value(cell.x+1, cell.y+1) * kernel[8];

  next_state[i] = applyActivation(sum);
}

fn index(cell: vec2u) -> u32 {
  // Supports grid wrap-around
  return (cell.y % u32(size))*u32(size)+(cell.x % u32(size));
}

fn value(x: u32, y: u32) -> f32 {
  return state[index(vec2(x,y))];
}

fn applyActivation(x: f32) -> f32 {
  ${activation}
}
`;

	return simulation;
}

export default continuous;
