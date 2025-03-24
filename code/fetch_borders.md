# Fetch Administrative Boundaries

A utility module to fetch administrative boundaries from OpenDataSoft's public API. This tool is particularly useful for geographic data analysis and visualization projects that require boundary data for different regions.

## Features

- Fetches boundary data for any continent and region
- Automatically converts JSON responses to geometric objects
- Handles both simple (Polygon) and complex (MultiPolygon) boundaries
- Built-in error handling with graceful fallbacks

## Quick Start

```python
from fetch_borders import fetch_boundaries

# Get South American boundaries (default)
boundaries = fetch_boundaries()

# Get European boundaries with custom parameters
europe_boundaries = fetch_boundaries(
    continent="Europe",
    region="Western Europe",
    limit=50
)
```

## Parameters

- `continent`: The target continent (default: "Americas")
- `region`: Specific region within the continent (default: "South America")
- `limit`: Maximum number of records to fetch (default: 100)

## Return Value

Returns a `GeoDataFrame` containing:
- Geometric shapes of administrative boundaries
- Empty GeoDataFrame if the request fails

## Dependencies

- requests
- geopandas
- shapely
- numpy

## Error Handling

The function handles API request failures gracefully:
- Returns an empty GeoDataFrame instead of raising exceptions
- Prints error messages for debugging purposes

## API Source

Uses the OpenDataSoft public API:
```
https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/world-administrative-boundaries
```

## Usage Tips

1. Consider API rate limits for production use
2. Adjust the `limit` parameter based on your needs
3. Use specific region names for more focused results
4. Check the returned GeoDataFrame's size to verify successful data retrieval 