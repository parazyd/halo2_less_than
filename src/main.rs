use halo2_proofs::arithmetic::Field;
use halo2_proofs::circuit::{floor_planner::V1, Layouter, Value};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{Advice, Circuit, Column, ConstraintSystem, Error};
use pasta_curves::{group::ff::PrimeField, Fp};

mod native_range_check;
use native_range_check::NativeRangeCheckChip;

mod less_than;
use less_than::{LessThanChip, LessThanConfig};

const WINDOW_SIZE: usize = 3;
const NUM_BITS: usize = 253;
const NUM_WINDOWS: usize = 85;

#[derive(Default)]
struct LessThanCircuit {
    a: Value<Fp>,
    b: Value<Fp>,
}

impl Circuit<Fp> for LessThanCircuit {
    type Config = (
        LessThanConfig<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>,
        Column<Advice>,
    );
    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        Self {
            a: Value::unknown(),
            b: Value::unknown(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let w = meta.advice_column();
        meta.enable_equality(w);

        let a = meta.advice_column();
        let b = meta.advice_column();
        let a_offset = meta.advice_column();
        let z1 = meta.advice_column();
        let z2 = meta.advice_column();

        let k_values_table = meta.lookup_table_column();

        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        (
            LessThanChip::<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>::configure(
                meta,
                a,
                b,
                a_offset,
                z1,
                z2,
                k_values_table,
            ),
            w,
        )
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let less_than_chip =
            LessThanChip::<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>::construct(config.0.clone());

        NativeRangeCheckChip::<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>::load_k_table(
            &mut layouter,
            config.0.k_values_table,
        )?;

        less_than_chip.witness_less_than(
            layouter.namespace(|| "a < b"),
            self.a,
            self.b,
            0,
            true,
        )?;

        Ok(())
    }
}

#[allow(non_snake_case)]
fn main() {
    let P_MINUS_1 = Fp::ZERO - Fp::ONE;

    // Maximum I'm able to check. Can we get more with
    // the same gadgets implementation?
    const MAX_253: Fp = Fp::from_raw([
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x1FFFFFFFFFFFFFFF,
    ]);

    // (a < b)
    let valid_pairs = [
        (Fp::ZERO, Fp::ONE),
        (Fp::from(u64::MAX), Fp::from(u64::MAX) + Fp::ONE),
        (Fp::from_u128(u128::MAX), Fp::from_u128(u128::MAX) + Fp::ONE),
        (MAX_253 - Fp::from(2), MAX_253 - Fp::ONE),
        (MAX_253 - Fp::ONE, MAX_253),
        (MAX_253, MAX_253 + Fp::ONE), // <-- Is this legit??
    ];

    for (a, b) in valid_pairs {
        let circuit = LessThanCircuit {
            a: Value::known(a),
            b: Value::known(b),
        };
        println!("[SHOULD PASS] {:?} < {:?}", a, b);
        let prover = MockProver::run(7, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    println!("");

    // (a < b)
    let invalid_pairs = [
        (Fp::ZERO, Fp::ZERO),
        (Fp::ONE, Fp::ZERO),
        (P_MINUS_1 - Fp::ONE, P_MINUS_1),
        (P_MINUS_1, P_MINUS_1),
        (P_MINUS_1, Fp::ZERO),
        (MAX_253, MAX_253),
        (MAX_253, Fp::ZERO),
        (MAX_253, Fp::ONE),
        (MAX_253 + Fp::ONE, Fp::ZERO),
        (MAX_253 + Fp::ONE, Fp::ONE),
        (MAX_253 + Fp::ONE, MAX_253 + Fp::ONE),
        (MAX_253 + Fp::ONE, MAX_253 + Fp::from(2)),
    ];

    for (a, b) in invalid_pairs {
        let circuit = LessThanCircuit {
            a: Value::known(a),
            b: Value::known(b),
        };
        println!("[SHOULD FAIL] {:?} < {:?}", a, b);
        let prover = MockProver::run(7, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
