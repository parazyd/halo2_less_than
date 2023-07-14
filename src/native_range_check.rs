use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, Value},
    pasta::{
        group::ff::{Field, PrimeFieldBits},
        pallas,
    },
    plonk,
    plonk::{Advice, Column, ConstraintSystem, Selector, TableColumn},
    poly::Rotation,
};

#[derive(Clone, Debug)]
pub struct NativeRangeCheckConfig<
    const WINDOW_SIZE: usize,
    const NUM_BITS: usize,
    const NUM_WINDOWS: usize,
> {
    pub z: Column<Advice>,
    pub s_rc: Selector,
    pub k_values_table: TableColumn,
}

#[derive(Clone, Debug)]
pub struct NativeRangeCheckChip<
    const WINDOW_SIZE: usize,
    const NUM_BITS: usize,
    const NUM_WINDOWS: usize,
> {
    config: NativeRangeCheckConfig<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>,
}

impl<const WINDOW_SIZE: usize, const NUM_BITS: usize, const NUM_WINDOWS: usize> Chip<pallas::Base>
    for NativeRangeCheckChip<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>
{
    type Config = NativeRangeCheckConfig<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<const WINDOW_SIZE: usize, const NUM_BITS: usize, const NUM_WINDOWS: usize>
    NativeRangeCheckChip<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>
{
    pub fn construct(config: NativeRangeCheckConfig<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS>) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        z: Column<Advice>,
        k_values_table: TableColumn,
    ) -> NativeRangeCheckConfig<WINDOW_SIZE, NUM_BITS, NUM_WINDOWS> {
        // Enable permutation on z column
        meta.enable_equality(z);

        let s_rc = meta.complex_selector();

        meta.lookup(|meta| {
            let s_rc = meta.query_selector(s_rc);
            let z_curr = meta.query_advice(z, Rotation::cur());
            let z_next = meta.query_advice(z, Rotation::next());

            //    z_next = (z_curr - k_i) / 2^K
            // => k_i = z_curr - (z_next * 2^K)
            vec![(
                s_rc * (z_curr - z_next * pallas::Base::from(1 << WINDOW_SIZE)),
                k_values_table,
            )]
        });

        NativeRangeCheckConfig {
            z,
            s_rc,
            k_values_table,
        }
    }

    /// `k_values_table` should be reused across different chips
    /// which is why we don't limit it to a specific instance.
    pub fn load_k_table(
        layouter: &mut impl Layouter<pallas::Base>,
        k_values_table: TableColumn,
    ) -> Result<(), plonk::Error> {
        layouter.assign_table(
            || format!("{} window table", WINDOW_SIZE),
            |mut table| {
                for index in 0..(1 << WINDOW_SIZE) {
                    table.assign_cell(
                        || format!("{} window assign", WINDOW_SIZE),
                        k_values_table,
                        index,
                        || Value::known(pallas::Base::from(index as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn decompose_value(value: &pallas::Base) -> Vec<[bool; WINDOW_SIZE]> {
        let padding = (WINDOW_SIZE - NUM_BITS % WINDOW_SIZE) % WINDOW_SIZE;

        let bits: Vec<bool> = value
            .to_le_bits()
            .into_iter()
            .take(NUM_BITS)
            .chain(std::iter::repeat(false).take(padding))
            .collect();
        assert_eq!(bits.len(), NUM_BITS + padding);

        bits.chunks_exact(WINDOW_SIZE)
            .map(|x| {
                let mut chunks = [false; WINDOW_SIZE];
                chunks.copy_from_slice(x);
                chunks
            })
            .collect()
    }

    pub fn decompose(
        &self,
        region: &mut Region<'_, pallas::Base>,
        z_0: AssignedCell<pallas::Base, pallas::Base>,
        offset: usize,
        strict: bool,
    ) -> Result<(), plonk::Error> {
        assert!(WINDOW_SIZE * NUM_WINDOWS < NUM_BITS + WINDOW_SIZE);

        // Enable selectors
        for index in 0..NUM_WINDOWS {
            self.config.s_rc.enable(region, index + offset)?;
        }

        let mut z_values: Vec<AssignedCell<pallas::Base, pallas::Base>> = vec![z_0.clone()];
        let mut z = z_0;
        let decomposed_chunks = z
            .value()
            .map(Self::decompose_value)
            .transpose_vec(NUM_WINDOWS);

        let two_pow_k_inverse = Value::known(
            pallas::Base::from(1 << WINDOW_SIZE as u64)
                .invert()
                .unwrap(),
        );

        for (i, chunk) in decomposed_chunks.iter().enumerate() {
            let z_next = {
                let z_curr = z.value().copied();
                let chunk_value = chunk.map(|c| {
                    pallas::Base::from(c.iter().rev().fold(0, |acc, c| (acc << 1) + *c as u64))
                });
                // z_next = (z_curr - k_i) / 2^K
                let z_next = (z_curr - chunk_value) * two_pow_k_inverse;
                region.assign_advice(
                    || format!("z_{}", i + offset + 1),
                    self.config.z,
                    i + offset + 1,
                    || z_next,
                )?
            };
            z_values.push(z_next.clone());
            z = z_next.clone();
        }

        assert!(z_values.len() == NUM_WINDOWS + 1);

        if strict {
            // Constrain the remaining bits to be zero
            region.constrain_constant(z_values.last().unwrap().cell(), pallas::Base::zero())?;
        }

        Ok(())
    }

    pub fn witness_range_check(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        value: Value<pallas::Base>,
        strict: bool,
    ) -> Result<(), plonk::Error> {
        layouter.assign_region(
            || format!("witness {}-bit native range check", NUM_BITS),
            |mut region: Region<'_, pallas::Base>| {
                let z_0 = region.assign_advice(|| "z_0", self.config.z, 0, || value)?;
                self.decompose(&mut region, z_0, 0, strict)?;
                Ok(())
            },
        )
    }

    pub fn copy_range_check(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        value: AssignedCell<pallas::Base, pallas::Base>,
        strict: bool,
    ) -> Result<(), plonk::Error> {
        layouter.assign_region(
            || format!("copy {}-bit native range check", NUM_BITS),
            |mut region: Region<'_, pallas::Base>| {
                let z_0 = value.copy_advice(|| "z_0", &mut region, self.config.z, 0)?;
                self.decompose(&mut region, z_0, 0, strict)?;
                Ok(())
            },
        )
    }
}
