# Mode: fix

Goal: turn the diagnosis into runnable code the user can drop into their pipeline.

## Rules
- Generate code only for fixes the user has confirmed they want. Never silently decide
  to drop a "leak" that might be real signal.
- Prefer fixes applied inside a scikit-learn Pipeline / ColumnTransformer so that
  imputation and scaling are fit on TRAIN ONLY — fitting on the full dataset is itself
  a leak, and this is the most common way users reintroduce one while "cleaning."
- For each generated block, add a one-line comment saying which finding it addresses.

## Typical outputs
- Drop ID-like and confirmed-leak columns.
- ColumnTransformer: median-impute + scale numerics, most-frequent-impute + encode cats.
- For high-cardinality categoricals: target or frequency encoding inside CV, not raw one-hot.
- For imbalance: class_weight or a resampler placed INSIDE the CV split, never before it.
