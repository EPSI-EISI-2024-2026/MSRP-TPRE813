from minio import Minio

def debugDF(df, rows: int = 20):
    if hasattr(df, 'columns') and hasattr(df, 'dtypes'):
        print(f"Columns and types: {[f'{col}: {dtype}' for col, dtype in zip(df.columns, df.dtypes)]}")
        print(f"Number of rows: {df.shape[0]}")
        print(f"First {rows} rows:")
        print(df.head(rows))
        print("Done printing head")
    elif hasattr(df, 'name') and hasattr(df, 'dtype'):
        print(f"Series name: {df.name}, dtype: {df.dtype}")
        print(f"Number of rows: {df.shape[0]}")
        print(f"First {rows} values:")
        print(df.head(rows))
        print("Done printing head (Series)")
    else:
        print("Object is neither a DataFrame nor a Series. Type:", type(df))
        print(df)

# MinIO file loader
def loadFile(client: Minio, bucket: str, filename: str) -> bytes:
    """
    Load a file from MinIO bucket.
    """
    try:
        data = client.get_object(bucket, filename)
        return data.read()
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None
