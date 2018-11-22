# PCA Operator

This IKATS operator implements a [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) (Principal Component Analysis). In a nutshell, it consists in reducing the dimention of a dataset, transforming *N* TS into *N_component < N* TS (same number of timestamps). TS data must be aligned.

## Input and parameters

This operator only takes one input of the functional type `ts_list`.

It also takes three optional parameter from the user:

- **Number of components**: The number of principal component to use (must be a positive integer)
- **FuncId pattern**: Uses Python string format to create a new funcid where `{pc_id}` is replaced by the id (from 1 to `n_components`) of each PC created (e.g. *PC_{pc_id}* produces TS named *PC_1*, *PC_2*, ...). Note that if `FuncID pattern` do not contains keyword `{pc_id}`, IKATS automatically adds `PC_{pc_id}` at the end of the *FuncId pattern*.
- **Table name**: The name given to the table containing variance explained (must be unique)

*Notes on `FuncID pattern`*: 
* If the `FuncID pattern` match with an already existing *fid* (example: `FuncID pattern = PC_{pc_id}`, but 'PC_1' already exist), Ikats raise an error.
* If `FuncID pattern` do not contain `{pc_id}`, add *PC_{pc_id}* at the end of the pattern.

*Note2*: Idem with arg `Table name`: raise error if the table already exists.

## Outputs

The operator has two outputs:
d}
- **TS list**: The resulting list of time series produced by the PCA (`n_components` TS). Note that these TS don't inherit from metadata (`metric`, etc) of the original TS.
- **Table**: A table containing explained variance (and cumulative explained variance) per Principal Component (PC)

### Warnings

- inputed TS must be aligned: same start/end date, same period (metadata `qual_ref_period`, or calculated if not provided)
- The process is sensitive to scaling, and there is no consensus as to how to best scale the data to obtain optimal results. In general, a [Z-Norm scaling](https://ikats.org/doc/operators/scale.html) is performed before processing a PCA.

### Implementation remarks: 

This version of PCA is designed as a wrapper of the python's [sklearn ](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) implementation. If the dataset provided is too important, Ikats switches automatically to a distributed [pyspark](https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.feature.PCA) implementation. Results fro the two implemntations can be different, du to the type of solver used.
