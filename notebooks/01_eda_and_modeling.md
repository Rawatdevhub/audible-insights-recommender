# EDA and modeling guide

Run the pipeline first:

```bash
python -m src.data_pipeline
python -m src.recommender --build
```

Suggested analysis questions:

1. How many rows are removed or merged during cleaning?
2. What is the rating distribution after removing sentinel `-1` values?
3. Which authors and genres dominate review volume?
4. Does review volume correlate with rating?
5. Which clusters are dominated by self-help, fiction, business, or science content?
6. Do recommendations remain diverse in author and title?

The processed file includes `cluster` after model training and is ready for a notebook or BI tool.
