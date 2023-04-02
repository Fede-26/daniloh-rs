// Inspired from mikeash's Fluid Simulation

// use daniloh_rs::{Fluid, diffuse};
use array2d::Array2D;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};

const WIDTH: usize = 200;
const HEIGHT: usize = WIDTH;
const ITER: u32 = 6;
const CLAMP_INPUT: f32 = 2.0;
const DT: f32 = 0.4;
const POWER: f32 = 250.0;

fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut default_options = WindowOptions::default();
    default_options.scale = minifb::Scale::X4;
    let mut window = Window::new("DANILOH - ESC to exit", WIDTH, HEIGHT, default_options)
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

    // Limit to max ~60 fps update rate
    // window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    // Save previous mouse position
    let mut prev_mouse = (0.5, 0.5);

    let mut fluid = FluidSquare::new(WIDTH, 0, 0, DT, ITER);
    println!("initialization done");
    while window.is_open() && !window.is_key_down(Key::Escape) {
        //add density where the mouse clicks
        if window.get_mouse_down(MouseButton::Left) {
            let (x, y) = window.get_mouse_pos(MouseMode::Clamp).unwrap();
            fluid.add_density(x as usize, y as usize, POWER);
            let x_clamped = (x-prev_mouse.0)/CLAMP_INPUT;
            let y_clamped = (y-prev_mouse.1)/ CLAMP_INPUT;
            fluid.add_velocity(x as usize, y as usize, x_clamped, y_clamped);
            prev_mouse = (x, y);
        }

        buffer = fluid
            .density
            .as_column_major()
            .iter()
            .map(|x| float_to_color(*x))
            .collect();
        
        fluid.step();
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
        println!("step done");
    }
}

//Struct that contains all the fluid properties and arrays
struct FluidSquare {
    //Fluid properties
    size: usize,
    dt: f32,
    diff: f32,
    visc: f32,
    iter: u32,

    //Fluid arrays
    s: Array2D<f32>,
    density: Array2D<f32>,

    //Velocity arrays
    Vx: Array2D<f32>,
    Vy: Array2D<f32>,
    Vx0: Array2D<f32>,
    Vy0: Array2D<f32>,
}

//FluidSquare methods
impl FluidSquare {
    fn new(size: usize, diffusion: u32, viscosity: u32, dt: f32, iter: u32) -> FluidSquare {
        FluidSquare {
            size: size,
            dt: dt,
            diff: diffusion as f32,
            visc: viscosity as f32,
            iter: iter,
            s: Array2D::filled_with(0.0, size, size),
            density: Array2D::filled_with(0.0, size, size),
            Vx: Array2D::filled_with(0.0, size, size),
            Vy: Array2D::filled_with(0.0, size, size),
            Vx0: Array2D::filled_with(0.0, size, size),
            Vy0: Array2D::filled_with(0.0, size, size),
        }
    }

    fn add_density(&mut self, x: usize, y: usize, amount: f32) {
        self.density[(x, y)] += amount;
    }

    fn add_velocity(&mut self, x: usize, y: usize, amount_x: f32, amount_y: f32) {
        self.Vx[(x, y)] += amount_x;
        self.Vy[(x, y)] += amount_y;
    }

    fn step(&mut self) {
        diffuse(1, &mut self.Vx0, &self.Vx, self.visc, self.dt, self.iter);
        diffuse(2, &mut self.Vy0, &self.Vy, self.visc, self.dt, self.iter);

        // self.project(self.Vx0, self.Vy0, self.Vx, self.Vy);
        project(
            &mut self.Vx0,
            &mut self.Vy0,
            &mut self.Vx,
            &mut self.Vy,
            self.iter,
        );

        advect(1, &mut self.Vx, &self.Vx0, &self.Vx0, &self.Vy0, self.dt);
        advect(2, &mut self.Vy, &self.Vy0, &self.Vx0, &self.Vy0, self.dt);

        // (self.Vx, self.Vy, self.Vx0, self.Vy0) = self.project(&mut self.Vx, &mut self.Vy, &mut self.Vx0, &mut self.Vy0);
        project(
            &mut self.Vx,
            &mut self.Vy,
            &mut self.Vx0,
            &mut self.Vy0,
            self.iter,
        );
        diffuse(0, &mut self.s, &self.density, self.diff, self.dt, self.iter);

        advect(0, &mut self.density, &self.s, &self.Vx, &self.Vy, self.dt);
    }
}
fn diffuse(b: usize, x: &mut Array2D<f32>, x0: &Array2D<f32>, diff: f32, dt: f32, iter: u32) {
    let a = dt * diff * (x.num_columns() - 2) as f32 * (x.num_rows() - 2) as f32;
    lin_solve(b, x, x0, a, 1.0 + 6.0 * a, iter);
}

fn project(
    veloc_x: &mut Array2D<f32>,
    veloc_y: &mut Array2D<f32>,
    p: &mut Array2D<f32>,
    div: &mut Array2D<f32>,
    iter: u32,
) {
    for i in 1..div.num_columns() - 1 {
        for j in 1..div.num_rows() - 1 {
            div[(i, j)] = -0.5
                * (veloc_x[(i + 1, j)] - veloc_x[(i - 1, j)] + veloc_y[(i, j + 1)]
                    - veloc_y[(i, j - 1)])
                / div.num_rows() as f32;
            p[(i, j)] = 0.0;
        }
    }

    set_bnd(0, div);
    set_bnd(0, p);
    lin_solve(0, p, div, 1.0, 6.0, iter);

    for i in 1..veloc_x.num_columns() - 1 {
        for j in 1..veloc_x.num_rows() - 1 {
            veloc_x[(i, j)] -= 0.5 * (p[(i + 1, j)] - p[(i - 1, j)]) * veloc_x.num_rows() as f32;
            veloc_y[(i, j)] -= 0.5 * (p[(i, j + 1)] - p[(i, j - 1)]) * veloc_x.num_rows() as f32;
        }
    }

    set_bnd(1, veloc_x);
    set_bnd(2, veloc_y);
}

//function advect that return only the array modified
fn advect(
    b: usize,
    d: &mut Array2D<f32>,
    d0: &Array2D<f32>,
    veloc_x: &Array2D<f32>,
    veloc_y: &Array2D<f32>,
    dt: f32,
) {
    let (mut i0, mut i1, mut j0, mut j1);
    let dtx = dt * (d.num_columns() - 2) as f32;
    let dty = dt * (d.num_rows() - 2) as f32;
    let (mut s0, mut s1, mut t0, mut t1);
    let (mut tmp1, mut tmp2, mut x, mut y): (f32, f32, f32, f32);
    let Nfloat = d.num_rows() as f32;
    let (mut ifloat, mut jfloat);

    for j in 1..d.num_rows() - 1 {
        jfloat = j as f32;
        for i in 1..d.num_columns() - 1 {
            ifloat = i as f32;
            tmp1 = dtx * veloc_x[(i, j)];
            tmp2 = dty * veloc_y[(i, j)];
            x = ifloat - tmp1;
            y = jfloat - tmp2;

            if x < 0.5 {
                x = 0.5
            };
            if x > Nfloat + 0.5 {
                x = Nfloat + 0.5
            };
            i0 = x.floor();
            i1 = i0 + 1.0;
            if y < 0.5 {
                y = 0.5
            };
            if y > Nfloat + 0.5 {
                y = Nfloat + 0.5
            };
            j0 = y.floor();
            j1 = j0 + 1.0;

            s1 = x - i0;
            s0 = 1.0 - s1;
            t1 = y - j0;
            t0 = 1.0 - t1;

            let i0i = (i0.floor() as usize).clamp(0, d.num_columns() - 1);
            let i1i = (i1.floor() as usize).clamp(0, d.num_columns() - 1);
            let j0i = (j0.floor() as usize).clamp(0, d.num_rows() - 1);
            let j1i = (j1.floor() as usize).clamp(0, d.num_rows() - 1);

            d[(i, j)] = s0 * (t0 * d0[(i0i, j0i)] + t1 * d0[(i0i, j1i)])
                + s1 * (t0 * d0[(i1i, j0i)] + t1 * d0[(i1i, j1i)]);
        }
    }
    set_bnd(b, d);
}

fn set_bnd(b: usize, x: &mut Array2D<f32>) {
    let n_col = x.num_columns();
    let n_rows = x.num_rows();
    for i in 1..n_col - 1 {
        x[(i, 0)] = if b == 2 { -x[(i, 1)] } else { x[(i, 1)] };
        x[(i, n_rows - 1)] = if b == 2 {
            -x[(i, n_rows - 2)]
        } else {
            x[(i, n_rows - 2)]
        };
    }

    for i in 1..n_rows - 1 {
        x[(0, i)] = if b == 1 { -x[(1, i)] } else { x[(1, i)] };
        x[(n_col - 1, i)] = if b == 1 {
            -x[(n_col - 2, i)]
        } else {
            x[(n_col - 2, i)]
        };
    }

    x[(0, 0)] = 0.5 * (x[(1, 0)] + x[(0, 1)]);
    x[(0, n_rows - 1)] = 0.5 * (x[(1, n_rows - 1)] + x[(0, n_rows - 2)]);
    x[(n_col - 1, 0)] = 0.5 * (x[(n_col - 2, 0)] + x[(n_col - 1, 1)]);
    x[(n_col - 1, n_rows - 1)] = 0.5 * (x[(n_col - 2, n_rows - 1)] + x[(n_col - 1, n_rows - 2)]);
}

fn lin_solve(b: usize, x: &mut Array2D<f32>, x0: &Array2D<f32>, a: f32, c: f32, iter: u32) {
    let c_recip = 1.0 / c;
    for _k in 0..iter {
        for i in 1..x.num_columns() - 1 {
            for j in 1..x.num_rows() - 1 {
                x[(i, j)] = (x0[(i, j)]
                    + a * (x[(i + 1, j)] + x[(i - 1, j)] + x[(i, j + 1)] + x[(i, j - 1)]))
                    * c_recip;
            }
        }
        set_bnd(b, x);
    }
}

fn float_to_color(f: f32) -> u32 {
    //convert a float that goes from 0 to 255 to a color rgb in u32
    // if f > 0.001 {
    //     println!("f is {}", f);
    // }
    // let i = (f*255.0/100.0 )as u32;
    let i = f as u32;
    i << 16 | i << 8 | i
}
