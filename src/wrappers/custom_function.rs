use paste::paste;
use std::{borrow::Borrow, cell::RefCell, ffi::c_int, rc::Rc};

use torch_sys::{at_shallow_clone, custom_function_ffi};

use crate::{TchError, Tensor};

// forward(input) -> output
pub trait ForwardFun = FnOnce(&Vec<Tensor>) -> Vec<Tensor> + 'static;
// backward(output_grad, inputs) -> input_grad
pub trait BackwardFun = FnOnce(Vec<Tensor>, &Vec<Tensor>) -> Vec<Tensor> + 'static;

pub trait CustomFunction: Sized + 'static {
    fn forward(&mut self) -> Vec<Tensor>;
    fn backward(&mut self, grad_output: Vec<Tensor>) -> Vec<Tensor>;
}

pub struct CustomFunctionProxy {
    forward: Option<Box<dyn ForwardFun>>,
    backward: Option<Box<dyn BackwardFun>>,
    use_args: Vec<Tensor>,
}
impl CustomFunction for CustomFunctionProxy {
    fn forward(&mut self) -> Vec<Tensor> {
        (self.forward.take().unwrap())(&self.use_args)
    }
    fn backward(&mut self, grad_output: Vec<Tensor>) -> Vec<Tensor> {
        (self.backward.take().unwrap())(grad_output, &self.use_args)
    }
}

/*
pub fn custom_function_fn<F, B, T: Borrow<Tensor>>(input: T, forward: F, backward: B) -> Tensor
where
    F: FnOnce(&Tensor) -> Tensor + 'static,
    // (grad, input)
    B: FnOnce(Tensor, &Tensor) -> Tensor + 'static,
{
    let t = {
        let forward = move |t: &Vec<Tensor>| vec![forward(&t[0])];
        let backward = move |grad_output: Vec<Tensor>, input: &Vec<Tensor>| {
            vec![backward(grad_output[0].shallow_clone(), &input[0])]
        };
        CustomFunctionProxy {
            forward: Some(Box::new(forward)),
            backward: Some(Box::new(backward)),
            use_args: vec![input.borrow().shallow_clone()],
        }
    };
    custom_function(t, &[input]).pop().unwrap()
}

pub fn custom_function_fn2<F, B, T: Borrow<Tensor>>(
    input: (T, T),
    forward: F,
    backward: B,
) -> Tensor
where
    F: FnOnce((&Tensor, &Tensor)) -> Tensor + 'static,
    B: FnOnce(Tensor, (&Tensor, &Tensor)) -> (Tensor, Tensor) + 'static,
{
    let t: CustomFunctionProxy = {
        let forward = move |t: &Vec<Tensor>| vec![forward((&t[0], &t[1]))];
        let backward = move |grad_output: Vec<Tensor>, input: &Vec<Tensor>| {
            let grad = backward(grad_output[0].shallow_clone(), (&input[0], &input[1]));
            vec![grad.0, grad.1]
        };
        CustomFunctionProxy {
            forward: Some(Box::new(forward)),
            backward: Some(Box::new(backward)),
            use_args: vec![input.0.borrow().shallow_clone(), input.1.borrow().shallow_clone()],
        }
    };
    custom_function(t, &[input.0, input.1]).pop().unwrap()
}
 */

macro_rules! type_as_literal {
    ($name:literal; $t: literal ) => {
        $name
    };
}
macro_rules! type_as_name {
    ($name:ty; $t: literal ) => {
        $name
    };
}
macro_rules! type_tuple_names {
    ($name:ty; $x:literal) => { $name };
    ($name:ty; $x:literal, $($y:literal),+) => { ($name, $(type_as_name!($name; $y)),* ) }
}
macro_rules! type_output_result {
    ($var:tt; $x:literal) => { $var.remove(0) };
    ($var:tt; $x:literal, $($y:literal),+) => {
        ($var.remove(0), $($var.remove(type_as_literal!(0; $y) )),*  )
    }
}
macro_rules! type_output_parameter {
    ($var:ident; $x:literal) => { $var[0].shallow_clone() };
    ($var:ident; $x:literal, $($y:literal),+) => {
        ($var[0].shallow_clone(), $($var[$y].shallow_clone()),* )
    }
}
// macro_rules! type_vec2tuple {
//     ($var:ident; $x:literal) => { &$var[0] };
//     ($var:ident; $x:literal, $($y:literal),+) => {
//         (&$var[0], $(&$var[$y]),* )
//     }
// }
macro_rules! type_tuple2vec {
    ($var:ident; $x:literal) => {
        vec![$var]
    };
    ($var:ident; $x:literal, $($y:tt),+) => {
        vec![ $var.0, $($var.$y),* ]
    }
}
macro_rules! type_tuple2array {
    ($var:ident; $x:literal) => {
        [$var]
    };
    ($var:ident; $x:literal, $($y:tt),+) => {
        [ $var.0, $($var.$y),* ]
    }
}
macro_rules! type_tuple2borrow_tensor {
    ($var:ident; $x:literal) => {
       $var.borrow().shallow_clone()
    };
    ($var:ident; $x:literal, $($y:tt),+) => {
        ( $var.0.borrow().shallow_clone(), $($var.$y.borrow().shallow_clone()),* )
    }
}

macro_rules! expand_custom_function {
    ($name:literal; $($input:tt),*; $($output:tt),* ) => {
        paste! {
            pub fn [<custom_function_fn $name>]<F, B, T: Borrow<Tensor>>(
                input: type_tuple_names!(T; $($input),* ),
                forward: F,
                backward: B,
            ) -> type_tuple_names!(Tensor; $($output),* )
            where
                F: FnOnce(&type_tuple_names!(Tensor; $($input),* ) ) -> type_tuple_names!(Tensor; $($output),* ) + 'static,
                B: FnOnce(type_tuple_names!(Tensor; $($output),* ), &type_tuple_names!(Tensor; $($input),* ) ) -> type_tuple_names!(Tensor; $($input),* ) + 'static,
            {

                struct CustomFunctionProxy {
                    forward: Option<Box<dyn FnOnce(& type_tuple_names!(Tensor; $($input),* ) ) -> type_tuple_names!(Tensor; $($output),* ) + 'static, >>,
                    backward: Option<Box<dyn FnOnce(type_tuple_names!(Tensor; $($output),* ), & type_tuple_names!(Tensor; $($input),* ) ) -> type_tuple_names!(Tensor; $($input),* ) + 'static,>>,
                    use_args: type_tuple_names!(Tensor; $($input),* ),
                }
                impl CustomFunction for CustomFunctionProxy {
                    fn forward(&mut self) -> Vec<Tensor> {
                        let forward = self.forward.take().unwrap();
                        let result = forward( &self.use_args );
                        type_tuple2vec!(result; $($output),* )
                    }
                    fn backward(&mut self, grad_output: Vec<Tensor>) -> Vec<Tensor> {
                        let backward = self.backward.take().unwrap();
                        let grad_output = type_output_parameter!(grad_output; $($output),* );
                        let grad = backward(grad_output, &self.use_args);
                        type_tuple2vec!(grad; $($input),*)
                    }
                }
                let ctx: CustomFunctionProxy = {
                    CustomFunctionProxy {
                        forward: Some(Box::new(forward)),
                        backward: Some(Box::new(backward)),
                        use_args: type_tuple2borrow_tensor!(input; $($input),*),
                    }
                };
                let mut result = custom_function(ctx, &type_tuple2array!(input; $($input),*) );
                type_output_result!(result; $($output),* )
            }
        }
    };
}

expand_custom_function!(1;0;0);
expand_custom_function!(2;0,1;0);
expand_custom_function!(3;0,1,2;0);
expand_custom_function!(4;0,1,2,3;0);
expand_custom_function!(5;0,1,2,3,4;0);
expand_custom_function!(12;0;0,1);
expand_custom_function!(22;0,1;0,1);
expand_custom_function!(32;0,1,2;0,1);
expand_custom_function!(42;0,1,2,3;0,1);
expand_custom_function!(52;0,1,2,3,4;0,1);
expand_custom_function!(13;0;0,1,2);
expand_custom_function!(23;0,1;0,1,2);
expand_custom_function!(33;0,1,2;0,1,2);
expand_custom_function!(43;0,1,2,3;0,1,2);
expand_custom_function!(53;0,1,2,3,4;0,1,2);
expand_custom_function!(14;0;0,1,2,3);
expand_custom_function!(24;0,1;0,1,2,3);
expand_custom_function!(34;0,1,2;0,1,2,3);
expand_custom_function!(44;0,1,2,3;0,1,2,3);
expand_custom_function!(54;0,1,2,3,4;0,1,2,3);

pub fn custom_function_array<F, B, T: Borrow<Tensor>>(
    input: &[T],
    forward: F,
    backward: B,
) -> Vec<Tensor>
where
    F: FnOnce(&Vec<Tensor>) -> Vec<Tensor> + 'static,
    B: FnOnce(Vec<Tensor>, &Vec<Tensor>) -> Vec<Tensor> + 'static,
{
    let ctx = CustomFunctionProxy {
        forward: Some(Box::new(forward)),
        backward: Some(Box::new(backward)),
        use_args: input.iter().map(|t| t.borrow().shallow_clone()).collect(),
    };
    custom_function(ctx, input)
}

pub fn custom_function<CTX, T: Borrow<Tensor>>(ctx: CTX, input: &[T]) -> Vec<Tensor>
where
    CTX: CustomFunction,
{
    let ctx = Rc::new(RefCell::new(ctx));
    let ctx_forward = ctx.clone();
    let ctx_backward = ctx.clone();

    let sz = input.len();
    debug_assert!(sz <= 10, "input.len() must be less than 10, but {}", input.len());
    let input_pointer = {
        let mut input_pointer = [std::ptr::null_mut(); 10];
        for i in 0..sz {
            input_pointer[i] = input[i].borrow().c_tensor;
        }
        input_pointer
    };

    let forward_result = Rc::new(RefCell::new(None));
    let forward_result1 = forward_result.clone();
    torch_sys::custom_function_ffi::custom_function_void(
        &input_pointer[0..sz],
        move || {
            let result = ctx_forward.borrow_mut().forward();
            let pointer = result
                .iter()
                .map(|t| t.c_tensor)
                // .map(|c_tensor| unsafe { at_shallow_clone(c_tensor) })
                .collect();
            forward_result.replace(Some(result));
            pointer
        },
        move |grads| {
            let grads = grads
                .iter()
                .map(|t| unsafe { at_shallow_clone(*t) })
                .map(|c_tensor| Tensor { c_tensor })
                .collect();
            let grads = ctx_backward.borrow_mut().backward(grads);
            grads.into_iter().map(|t| unsafe { at_shallow_clone(t.c_tensor) }).collect()
        },
    );
    forward_result1.take().unwrap()
}

// run backward batch
// parameters:
//   tensors: which tensors are backwards
//   grad_tensors: derive from tensors if None
//   input: calc all if None
pub fn run_backward_batch<T1, T2, T3>(
    tensors: &[T1],
    grad_tensors: &[T2],
    inputs: &[T3],
    keep_graph: bool,
    create_graph: bool,
) where
    T1: Borrow<Tensor>,
    T2: Borrow<Tensor>,
    T3: Borrow<Tensor>,
{
    f_run_backward_batch(tensors, grad_tensors, inputs, keep_graph, create_graph).unwrap()
}
pub fn f_run_backward_batch<T1, T2, T3>(
    tensors: &[T1],
    grad_tensors: &[T2],
    inputs: &[T3],
    keep_graph: bool,
    create_graph: bool,
) -> Result<(), TchError>
where
    T1: Borrow<Tensor>,
    T2: Borrow<Tensor>,
    T3: Borrow<Tensor>,
{
    let tensors: Vec<_> = tensors.iter().map(|x| x.borrow().c_tensor).collect();
    let grad_tensors: Vec<_> = grad_tensors.iter().map(|x| x.borrow().c_tensor).collect();
    let inputs: Vec<_> = inputs.iter().map(|x| x.borrow().c_tensor).collect();

    unsafe_torch_err!(custom_function_ffi::run_backward_batch(
        tensors.as_ptr(),
        tensors.len() as c_int,
        grad_tensors.as_ptr(),
        grad_tensors.len() as c_int,
        inputs.as_ptr(),
        inputs.len() as c_int,
        keep_graph as c_int,
        create_graph as c_int,
    ));
    Ok(())
}

pub fn is_need_backward<T>(tensors: &[T]) -> bool
where
    T: Borrow<Tensor>,
{
    unsafe {
        custom_function_ffi::is_grad_enabled() && tensors.iter().any(|t| t.borrow().requires_grad())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::{kind::FLOAT_CPU, no_grad, Device, Kind, Reduction, Tensor};

    #[test]
    fn test_base() -> anyhow::Result<()> {
        // y = x^2 + sin x
        // y' = 2x + cos x

        // target = [2,1]^T

        let mut x = Tensor::arange(2 * 3, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);
        let target = Tensor::from_slice(&[2., 1.])
            .to_kind(Kind::Float)
            .reshape(&[2, 1])
            .to_device(Device::Cpu);
        let expect_grad = vec![-0.6666667, -0.13423723, 3.4754982, 13.595691, 34.878536, 78.982216];

        let y = &x * &x + &x.sin();
        let loss = y.mse_loss(&target, Reduction::Mean);
        assert_eq!(135.421142578125, loss.double_value(&[]));

        loss.backward();
        let grad: Vec<f32> = x.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_grad);

        x.zero_grad();

        println!("before custom function");
        let y = custom_function_fn1(
            &x,
            |x| {
                let result = x * x + x.sin();
                println!("forward: require grad:{}->{}", x.requires_grad(), result.requires_grad());
                result
            },
            |ref grad, x| {
                println!("backward call!");
                grad * (2. * x + x.cos())
            },
        );
        let loss = y.mse_loss(&target, Reduction::Mean);
        assert_eq!(135.421142578125, loss.double_value(&[]));

        println!("y={:?}, require grad:{}", y, y.requires_grad());
        loss.backward();
        let grad: Vec<f32> = x.grad().view(-1).try_into()?;
        grad.iter().zip(expect_grad.iter()).for_each(|(a, b)| assert_relative_eq!(a, b));

        Ok(())
    }

    #[test]
    fn test_base_2() -> anyhow::Result<()> {
        // y = x^2 + sin v
        // y' = 2x + cos v

        // target = [2,1]^T

        let mut x = Tensor::arange(2 * 3, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);
        let mut v = Tensor::arange(2 * 3, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);
        let target = Tensor::from_slice(&[2., 1.])
            .to_kind(Kind::Float)
            .reshape(&[2, 1])
            .to_device(Device::Cpu);
        let expect_x_grad = vec![-0.0, -0.10568603, 3.8790634, 16.28224, 37.98186, 76.80359];
        let expect_v_grad =
            vec![-0.6666667, -0.028551204, -0.403565, -2.6865494, -3.1033251, 2.1786275];

        let y = &x * &x + &v.sin();
        let loss = y.mse_loss(&target, Reduction::Mean);
        assert_eq!(135.421142578125, loss.double_value(&[]));

        loss.backward();
        let grad: Vec<f32> = x.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_x_grad);

        let grad: Vec<f32> = v.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_v_grad);

        x.zero_grad();
        v.zero_grad();

        println!("before custom function");
        let y = custom_function_fn2(
            (&x, &v),
            |(x, v)| {
                let result = x * x + v.sin();
                println!("forward: require grad:{}->{}", x.requires_grad(), result.requires_grad());
                result
            },
            |ref grad, (x, v)| {
                println!("backward call");
                let x_grad = grad * 2 * x;
                let v_grad = grad * v.cos();
                (x_grad, v_grad)
            },
        );
        let loss = y.mse_loss(&target, Reduction::Mean);
        assert_eq!(135.421142578125, loss.double_value(&[]));

        println!("y={:?}, require grad:{}", y, y.requires_grad());
        loss.backward();
        let grad: Vec<f32> = x.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_x_grad);
        grad.iter().zip(expect_x_grad.iter()).for_each(|(a, b)| assert_relative_eq!(a, b));

        let grad: Vec<f32> = v.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_v_grad);

        Ok(())
    }

    #[test]
    fn test_base_3() -> anyhow::Result<()> {
        // x = a^2 + sin b
        // y = 2a + cos c
        // z = x*d + y
        // target = [2,1]^T

        // x' = 2a + cos b
        // y' = 2 -sin c
        // z' = x + d + y'
        let mut a = Tensor::arange(2 * 3, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);
        let mut b =
            Tensor::arange_start(1, 2 * 3 + 1, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);
        let mut c =
            Tensor::arange_start(2, 2 * 3 + 2, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);
        let mut d =
            Tensor::arange_start(3, 2 * 3 + 3, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);

        let expect_a_grad = vec![0.07217741, 22.157326, 161.71436, 693.4094, 2189.4556, 5672.175];
        let expect_b_grad =
            vec![0.058496434, -3.6882806, -36.385452, -71.56462, 74.956215, 531.34186];
        let expect_c_grad =
            vec![-0.032815367, -0.31268418, 5.562992, 17.498081, 10.547721, -45.445644];
        let expect_d_grad = vec![0.030367598, 4.2304926, 30.439932, 150.4187, 567.78906, 1709.9938];

        let target = Tensor::from_slice(&[2., 1.])
            .to_kind(Kind::Float)
            .reshape(&[2, 1])
            .to_device(Device::Cpu);

        let x = &a * &a + b.sin();
        let y = 2. * &a + c.cos();
        let z = &x * &d + &y;

        let loss = z.mse_loss(&target, Reduction::Mean);
        assert_eq!(9902.7109375, loss.double_value(&[]));

        loss.backward();
        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_a_grad);

        let grad: Vec<f32> = b.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_b_grad);
        let grad: Vec<f32> = c.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_c_grad);
        let grad: Vec<f32> = d.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_d_grad);

        a.zero_grad();
        b.zero_grad();
        c.zero_grad();
        d.zero_grad();

        let b1 = custom_function_fn1(&b, |b| b.sin(), |ref grad, b| grad * b.cos());
        let (x, y) = custom_function_fn32(
            (&a, &b1, &c),
            |(a, b1, c)| {
                let x = a * a + b1;
                let y = 2. * a + c.cos();
                (x, y)
            },
            |(ref x_grad, ref y_grad), (a, _b1, c)| {
                let a_grad = 2 * y_grad + 2 * a * x_grad;
                let b1_grad = x_grad.shallow_clone();
                let c_grad = -y_grad * c.sin();
                (a_grad, b1_grad, c_grad)
            },
        );
        let z = custom_function_fn3(
            (&x, &d, &y),
            |(x, d, y)| x * d + y,
            |ref z_grad, (x, d, _y)| {
                let x_grad = z_grad * d;
                let d_grad = z_grad * x;
                let y_grad = z_grad.shallow_clone();
                (x_grad, d_grad, y_grad)
            },
        );
        let loss = z.mse_loss(&target, Reduction::Mean);
        assert_eq!(9902.7109375, loss.double_value(&[]));

        loss.backward();
        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_a_grad);

        let grad: Vec<f32> = b.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_b_grad);
        let grad: Vec<f32> = c.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_c_grad);
        let grad: Vec<f32> = d.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_d_grad);

        Ok(())
    }

    #[test]
    fn test_no_grad() -> anyhow::Result<()> {
        // x = a * a
        let mut a = Tensor::arange(2 * 3, FLOAT_CPU).reshape(&[2, 3]).set_requires_grad(true);
        let expect_a_grad = vec![-0.0, -0.6666667, 2.6666667, 16.0, 40.0, 80.0];
        let expect_a_grad_custom = vec![0.0, -1.0, 4.0, 24.0, 60.0, 120.0];

        let target = Tensor::from_slice(&[2., 1.])
            .to_kind(Kind::Float)
            .reshape(&[2, 1])
            .to_device(Device::Cpu);

        let x = &a * &a;
        let loss = x.mse_loss(&target, Reduction::Mean);
        assert_eq!(145.6666717529297, loss.double_value(&[]));

        loss.backward();
        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_a_grad);

        a.zero_grad();
        let x = custom_function_fn1(&a, |a| a * a, |ref grad, a| grad * 3. * a);
        let loss = x.mse_loss(&target, Reduction::Mean);
        assert_eq!(145.6666717529297, loss.double_value(&[]));

        loss.backward();
        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_a_grad_custom);

        a.zero_grad();

        let x = no_grad(|| &a * &a);
        let loss = x.mse_loss(&target, Reduction::Mean);
        assert_eq!(145.6666717529297, loss.double_value(&[]));
        assert_eq!(false, loss.requires_grad());

        // custom grad
        a.zero_grad();
        let x = no_grad(|| custom_function_fn1(&a, |a| a * a, |ref grad, a| grad * 3. * a));
        let loss = x.mse_loss(&target, Reduction::Mean);
        assert_eq!(145.6666717529297, loss.double_value(&[]));
        assert_eq!(false, loss.requires_grad());

        let a = a.detach();
        let x = custom_function_fn1(&a, |a| a * a, |ref grad, a| grad * 3. * a);
        let loss = x.mse_loss(&target, Reduction::Mean);
        assert_eq!(145.6666717529297, loss.double_value(&[]));
        assert_eq!(false, loss.requires_grad());

        Ok(())
    }

    #[test]
    fn test_backward_batch() -> anyhow::Result<()> {
        let mut hidden = Tensor::arange_start(1, 1 * 2 + 1, FLOAT_CPU).reshape(&[1, 2]);
        let _ = hidden.shallow_clone().requires_grad_(true);
        let mut a =
            Tensor::arange_start(2, 1 * 2 + 2, FLOAT_CPU).reshape(&[1, 2]).set_requires_grad(true);
        let target = &Tensor::from_slice(&[3, 4]).to_kind(Kind::Float).reshape(&[1, 2]);

        let expect_a_grad = vec![4.0, 168.0];
        let expect_hidden_grad = vec![4.0, 126.0];
        let expect_loss = 98.5;

        // x = a*a * hidden
        // x' = 2a*hidden + a*a * hidden'
        let x = &a * &a * &hidden;
        let loss = x.mse_loss(target, Reduction::Mean);
        assert_eq!(expect_loss, loss.double_value(&[]));
        loss.backward();

        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_a_grad);

        let grad: Vec<f32> = hidden.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_hidden_grad);

        // backward grad manually
        hidden.zero_grad();
        a.zero_grad();

        let x = &a * &a * &hidden;
        let loss = no_grad(|| &x - target);
        x.backward_with_grad(&loss);

        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_a_grad);

        let grad: Vec<f32> = hidden.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_hidden_grad);

        // backward grad batch
        hidden.zero_grad();
        a.zero_grad();

        let x = &a * &a * &hidden;
        let grad_tensors = no_grad(|| &x - target);
        run_backward_batch::<_, _, &Tensor>(&[&x], &[&grad_tensors], &[], false, false);

        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_a_grad);

        let grad: Vec<f32> = hidden.grad().view(-1).try_into()?;
        assert_eq!(grad, expect_hidden_grad);

        // backward grad batch for multi tensors
        let x = &a * &a * &hidden;
        let grad_x = no_grad(|| &x - target);

        let y = 2 * &a + &hidden;
        let grad_y = no_grad(|| &y - target);

        run_backward_batch::<_, _, &Tensor>(&[&x, &y], &[&grad_x, &grad_y], &[], false, false);

        let grad: Vec<f32> = a.grad().view(-1).try_into()?;
        assert_eq!(grad, [12.0, 344.0]);

        let grad: Vec<f32> = hidden.grad().view(-1).try_into()?;
        assert_eq!(grad, [10.0, 256.0]);

        Ok(())
    }
}
