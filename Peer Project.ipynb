{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": " <a href=\"https://www.bigdatauniversity.com\"><img src = \"https://ibm.box.com/shared/static/ugcqz6ohbvff804xp84y4kqnvvk3bq1g.png\" width = 300, align = \"center\"></a>\n\n<h1 align=center><font size = 5>Data Analysis with Python</font></h1>"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# House Sales in King County, USA"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015."
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "<b>id</b> : A notation for a house\n\n<b> date</b>: Date house was sold\n\n\n<b>price</b>: Price is prediction target\n\n\n<b>bedrooms</b>: Number of bedrooms\n\n\n<b>bathrooms</b>: Number of bathrooms\n\n<b>sqft_living</b>: Square footage of the home\n\n<b>sqft_lot</b>: Square footage of the lot\n\n\n<b>floors</b> :Total floors (levels) in house\n\n\n<b>waterfront</b> :House which has a view to a waterfront\n\n\n<b>view</b>: Has been viewed\n\n\n<b>condition</b> :How good the condition is overall\n\n<b>grade</b>: overall grade given to the housing unit, based on King County grading system\n\n\n<b>sqft_above</b> : Square footage of house apart from basement\n\n\n<b>sqft_basement</b>: Square footage of the basement\n\n<b>yr_built</b> : Built Year\n\n\n<b>yr_renovated</b> : Year when house was renovated\n\n<b>zipcode</b>: Zip code\n\n\n<b>lat</b>: Latitude coordinate\n\n<b>long</b>: Longitude coordinate\n\n<b>sqft_living15</b> : Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area\n\n\n<b>sqft_lot15</b> : LotSize area in 2015(implies-- some renovations)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "You will require the following libraries: "
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": "import pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport seaborn as sns\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler,PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\n%matplotlib inline"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Module 1: Importing Data Sets "
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": " Load the csv:  "
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [],
            "source": "file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'\ndf=pd.read_csv(file_name)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\nWe use the method <code>head</code> to display the first 5 columns of the dataframe."
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Unnamed: 0</th>\n      <th>id</th>\n      <th>date</th>\n      <th>price</th>\n      <th>bedrooms</th>\n      <th>bathrooms</th>\n      <th>sqft_living</th>\n      <th>sqft_lot</th>\n      <th>floors</th>\n      <th>waterfront</th>\n      <th>...</th>\n      <th>grade</th>\n      <th>sqft_above</th>\n      <th>sqft_basement</th>\n      <th>yr_built</th>\n      <th>yr_renovated</th>\n      <th>zipcode</th>\n      <th>lat</th>\n      <th>long</th>\n      <th>sqft_living15</th>\n      <th>sqft_lot15</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0</td>\n      <td>7129300520</td>\n      <td>20141013T000000</td>\n      <td>221900.0</td>\n      <td>3.0</td>\n      <td>1.00</td>\n      <td>1180</td>\n      <td>5650</td>\n      <td>1.0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>7</td>\n      <td>1180</td>\n      <td>0</td>\n      <td>1955</td>\n      <td>0</td>\n      <td>98178</td>\n      <td>47.5112</td>\n      <td>-122.257</td>\n      <td>1340</td>\n      <td>5650</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>1</td>\n      <td>6414100192</td>\n      <td>20141209T000000</td>\n      <td>538000.0</td>\n      <td>3.0</td>\n      <td>2.25</td>\n      <td>2570</td>\n      <td>7242</td>\n      <td>2.0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>7</td>\n      <td>2170</td>\n      <td>400</td>\n      <td>1951</td>\n      <td>1991</td>\n      <td>98125</td>\n      <td>47.7210</td>\n      <td>-122.319</td>\n      <td>1690</td>\n      <td>7639</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>2</td>\n      <td>5631500400</td>\n      <td>20150225T000000</td>\n      <td>180000.0</td>\n      <td>2.0</td>\n      <td>1.00</td>\n      <td>770</td>\n      <td>10000</td>\n      <td>1.0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>6</td>\n      <td>770</td>\n      <td>0</td>\n      <td>1933</td>\n      <td>0</td>\n      <td>98028</td>\n      <td>47.7379</td>\n      <td>-122.233</td>\n      <td>2720</td>\n      <td>8062</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>3</td>\n      <td>2487200875</td>\n      <td>20141209T000000</td>\n      <td>604000.0</td>\n      <td>4.0</td>\n      <td>3.00</td>\n      <td>1960</td>\n      <td>5000</td>\n      <td>1.0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>7</td>\n      <td>1050</td>\n      <td>910</td>\n      <td>1965</td>\n      <td>0</td>\n      <td>98136</td>\n      <td>47.5208</td>\n      <td>-122.393</td>\n      <td>1360</td>\n      <td>5000</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>4</td>\n      <td>1954400510</td>\n      <td>20150218T000000</td>\n      <td>510000.0</td>\n      <td>3.0</td>\n      <td>2.00</td>\n      <td>1680</td>\n      <td>8080</td>\n      <td>1.0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>8</td>\n      <td>1680</td>\n      <td>0</td>\n      <td>1987</td>\n      <td>0</td>\n      <td>98074</td>\n      <td>47.6168</td>\n      <td>-122.045</td>\n      <td>1800</td>\n      <td>7503</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 22 columns</p>\n</div>",
                        "text/plain": "   Unnamed: 0          id             date     price  bedrooms  bathrooms  \\\n0           0  7129300520  20141013T000000  221900.0       3.0       1.00   \n1           1  6414100192  20141209T000000  538000.0       3.0       2.25   \n2           2  5631500400  20150225T000000  180000.0       2.0       1.00   \n3           3  2487200875  20141209T000000  604000.0       4.0       3.00   \n4           4  1954400510  20150218T000000  510000.0       3.0       2.00   \n\n   sqft_living  sqft_lot  floors  waterfront  ...  grade  sqft_above  \\\n0         1180      5650     1.0           0  ...      7        1180   \n1         2570      7242     2.0           0  ...      7        2170   \n2          770     10000     1.0           0  ...      6         770   \n3         1960      5000     1.0           0  ...      7        1050   \n4         1680      8080     1.0           0  ...      8        1680   \n\n   sqft_basement  yr_built  yr_renovated  zipcode      lat     long  \\\n0              0      1955             0    98178  47.5112 -122.257   \n1            400      1951          1991    98125  47.7210 -122.319   \n2              0      1933             0    98028  47.7379 -122.233   \n3            910      1965             0    98136  47.5208 -122.393   \n4              0      1987             0    98074  47.6168 -122.045   \n\n   sqft_living15  sqft_lot15  \n0           1340        5650  \n1           1690        7639  \n2           2720        8062  \n3           1360        5000  \n4           1800        7503  \n\n[5 rows x 22 columns]"
                    },
                    "execution_count": 3,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df.head()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 1 \nDisplay the data types of each column using the attribute dtype, then take a screenshot and submit it, include your code in the image. "
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "Unnamed: 0         int64\nid                 int64\ndate              object\nprice            float64\nbedrooms         float64\nbathrooms        float64\nsqft_living        int64\nsqft_lot           int64\nfloors           float64\nwaterfront         int64\nview               int64\ncondition          int64\ngrade              int64\nsqft_above         int64\nsqft_basement      int64\nyr_built           int64\nyr_renovated       int64\nzipcode            int64\nlat              float64\nlong             float64\nsqft_living15      int64\nsqft_lot15         int64\ndtype: object"
                    },
                    "execution_count": 4,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df.dtypes"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "We use the method describe to obtain a statistical summary of the dataframe."
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                },
                "scrolled": false
            },
            "outputs": [
                {
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Unnamed: 0</th>\n      <th>id</th>\n      <th>price</th>\n      <th>bedrooms</th>\n      <th>bathrooms</th>\n      <th>sqft_living</th>\n      <th>sqft_lot</th>\n      <th>floors</th>\n      <th>waterfront</th>\n      <th>view</th>\n      <th>...</th>\n      <th>grade</th>\n      <th>sqft_above</th>\n      <th>sqft_basement</th>\n      <th>yr_built</th>\n      <th>yr_renovated</th>\n      <th>zipcode</th>\n      <th>lat</th>\n      <th>long</th>\n      <th>sqft_living15</th>\n      <th>sqft_lot15</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>21613.00000</td>\n      <td>2.161300e+04</td>\n      <td>2.161300e+04</td>\n      <td>21600.000000</td>\n      <td>21603.000000</td>\n      <td>21613.000000</td>\n      <td>2.161300e+04</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>...</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>10806.00000</td>\n      <td>4.580302e+09</td>\n      <td>5.400881e+05</td>\n      <td>3.372870</td>\n      <td>2.115736</td>\n      <td>2079.899736</td>\n      <td>1.510697e+04</td>\n      <td>1.494309</td>\n      <td>0.007542</td>\n      <td>0.234303</td>\n      <td>...</td>\n      <td>7.656873</td>\n      <td>1788.390691</td>\n      <td>291.509045</td>\n      <td>1971.005136</td>\n      <td>84.402258</td>\n      <td>98077.939805</td>\n      <td>47.560053</td>\n      <td>-122.213896</td>\n      <td>1986.552492</td>\n      <td>12768.455652</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>6239.28002</td>\n      <td>2.876566e+09</td>\n      <td>3.671272e+05</td>\n      <td>0.926657</td>\n      <td>0.768996</td>\n      <td>918.440897</td>\n      <td>4.142051e+04</td>\n      <td>0.539989</td>\n      <td>0.086517</td>\n      <td>0.766318</td>\n      <td>...</td>\n      <td>1.175459</td>\n      <td>828.090978</td>\n      <td>442.575043</td>\n      <td>29.373411</td>\n      <td>401.679240</td>\n      <td>53.505026</td>\n      <td>0.138564</td>\n      <td>0.140828</td>\n      <td>685.391304</td>\n      <td>27304.179631</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>0.00000</td>\n      <td>1.000102e+06</td>\n      <td>7.500000e+04</td>\n      <td>1.000000</td>\n      <td>0.500000</td>\n      <td>290.000000</td>\n      <td>5.200000e+02</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>...</td>\n      <td>1.000000</td>\n      <td>290.000000</td>\n      <td>0.000000</td>\n      <td>1900.000000</td>\n      <td>0.000000</td>\n      <td>98001.000000</td>\n      <td>47.155900</td>\n      <td>-122.519000</td>\n      <td>399.000000</td>\n      <td>651.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>5403.00000</td>\n      <td>2.123049e+09</td>\n      <td>3.219500e+05</td>\n      <td>3.000000</td>\n      <td>1.750000</td>\n      <td>1427.000000</td>\n      <td>5.040000e+03</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>...</td>\n      <td>7.000000</td>\n      <td>1190.000000</td>\n      <td>0.000000</td>\n      <td>1951.000000</td>\n      <td>0.000000</td>\n      <td>98033.000000</td>\n      <td>47.471000</td>\n      <td>-122.328000</td>\n      <td>1490.000000</td>\n      <td>5100.000000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>10806.00000</td>\n      <td>3.904930e+09</td>\n      <td>4.500000e+05</td>\n      <td>3.000000</td>\n      <td>2.250000</td>\n      <td>1910.000000</td>\n      <td>7.618000e+03</td>\n      <td>1.500000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>...</td>\n      <td>7.000000</td>\n      <td>1560.000000</td>\n      <td>0.000000</td>\n      <td>1975.000000</td>\n      <td>0.000000</td>\n      <td>98065.000000</td>\n      <td>47.571800</td>\n      <td>-122.230000</td>\n      <td>1840.000000</td>\n      <td>7620.000000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>16209.00000</td>\n      <td>7.308900e+09</td>\n      <td>6.450000e+05</td>\n      <td>4.000000</td>\n      <td>2.500000</td>\n      <td>2550.000000</td>\n      <td>1.068800e+04</td>\n      <td>2.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>...</td>\n      <td>8.000000</td>\n      <td>2210.000000</td>\n      <td>560.000000</td>\n      <td>1997.000000</td>\n      <td>0.000000</td>\n      <td>98118.000000</td>\n      <td>47.678000</td>\n      <td>-122.125000</td>\n      <td>2360.000000</td>\n      <td>10083.000000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>21612.00000</td>\n      <td>9.900000e+09</td>\n      <td>7.700000e+06</td>\n      <td>33.000000</td>\n      <td>8.000000</td>\n      <td>13540.000000</td>\n      <td>1.651359e+06</td>\n      <td>3.500000</td>\n      <td>1.000000</td>\n      <td>4.000000</td>\n      <td>...</td>\n      <td>13.000000</td>\n      <td>9410.000000</td>\n      <td>4820.000000</td>\n      <td>2015.000000</td>\n      <td>2015.000000</td>\n      <td>98199.000000</td>\n      <td>47.777600</td>\n      <td>-121.315000</td>\n      <td>6210.000000</td>\n      <td>871200.000000</td>\n    </tr>\n  </tbody>\n</table>\n<p>8 rows \u00d7 21 columns</p>\n</div>",
                        "text/plain": "        Unnamed: 0            id         price      bedrooms     bathrooms  \\\ncount  21613.00000  2.161300e+04  2.161300e+04  21600.000000  21603.000000   \nmean   10806.00000  4.580302e+09  5.400881e+05      3.372870      2.115736   \nstd     6239.28002  2.876566e+09  3.671272e+05      0.926657      0.768996   \nmin        0.00000  1.000102e+06  7.500000e+04      1.000000      0.500000   \n25%     5403.00000  2.123049e+09  3.219500e+05      3.000000      1.750000   \n50%    10806.00000  3.904930e+09  4.500000e+05      3.000000      2.250000   \n75%    16209.00000  7.308900e+09  6.450000e+05      4.000000      2.500000   \nmax    21612.00000  9.900000e+09  7.700000e+06     33.000000      8.000000   \n\n        sqft_living      sqft_lot        floors    waterfront          view  \\\ncount  21613.000000  2.161300e+04  21613.000000  21613.000000  21613.000000   \nmean    2079.899736  1.510697e+04      1.494309      0.007542      0.234303   \nstd      918.440897  4.142051e+04      0.539989      0.086517      0.766318   \nmin      290.000000  5.200000e+02      1.000000      0.000000      0.000000   \n25%     1427.000000  5.040000e+03      1.000000      0.000000      0.000000   \n50%     1910.000000  7.618000e+03      1.500000      0.000000      0.000000   \n75%     2550.000000  1.068800e+04      2.000000      0.000000      0.000000   \nmax    13540.000000  1.651359e+06      3.500000      1.000000      4.000000   \n\n       ...         grade    sqft_above  sqft_basement      yr_built  \\\ncount  ...  21613.000000  21613.000000   21613.000000  21613.000000   \nmean   ...      7.656873   1788.390691     291.509045   1971.005136   \nstd    ...      1.175459    828.090978     442.575043     29.373411   \nmin    ...      1.000000    290.000000       0.000000   1900.000000   \n25%    ...      7.000000   1190.000000       0.000000   1951.000000   \n50%    ...      7.000000   1560.000000       0.000000   1975.000000   \n75%    ...      8.000000   2210.000000     560.000000   1997.000000   \nmax    ...     13.000000   9410.000000    4820.000000   2015.000000   \n\n       yr_renovated       zipcode           lat          long  sqft_living15  \\\ncount  21613.000000  21613.000000  21613.000000  21613.000000   21613.000000   \nmean      84.402258  98077.939805     47.560053   -122.213896    1986.552492   \nstd      401.679240     53.505026      0.138564      0.140828     685.391304   \nmin        0.000000  98001.000000     47.155900   -122.519000     399.000000   \n25%        0.000000  98033.000000     47.471000   -122.328000    1490.000000   \n50%        0.000000  98065.000000     47.571800   -122.230000    1840.000000   \n75%        0.000000  98118.000000     47.678000   -122.125000    2360.000000   \nmax     2015.000000  98199.000000     47.777600   -121.315000    6210.000000   \n\n          sqft_lot15  \ncount   21613.000000  \nmean    12768.455652  \nstd     27304.179631  \nmin       651.000000  \n25%      5100.000000  \n50%      7620.000000  \n75%     10083.000000  \nmax    871200.000000  \n\n[8 rows x 21 columns]"
                    },
                    "execution_count": 5,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df.describe()"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": "df.info()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Module 2: Data Wrangling"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 2 \nDrop the columns <code>\"id\"</code>  and <code>\"Unnamed: 0\"</code> from axis 1 using the method <code>drop()</code>, then use the method <code>describe()</code> to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the <code>inplace</code> parameter is set to <code>True</code>"
        },
        {
            "cell_type": "code",
            "execution_count": 10,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>price</th>\n      <th>bedrooms</th>\n      <th>bathrooms</th>\n      <th>sqft_living</th>\n      <th>sqft_lot</th>\n      <th>floors</th>\n      <th>waterfront</th>\n      <th>view</th>\n      <th>condition</th>\n      <th>grade</th>\n      <th>sqft_above</th>\n      <th>sqft_basement</th>\n      <th>yr_built</th>\n      <th>yr_renovated</th>\n      <th>zipcode</th>\n      <th>lat</th>\n      <th>long</th>\n      <th>sqft_living15</th>\n      <th>sqft_lot15</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>2.161300e+04</td>\n      <td>21600.000000</td>\n      <td>21603.000000</td>\n      <td>21613.000000</td>\n      <td>2.161300e+04</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n      <td>21613.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>5.400881e+05</td>\n      <td>3.372870</td>\n      <td>2.115736</td>\n      <td>2079.899736</td>\n      <td>1.510697e+04</td>\n      <td>1.494309</td>\n      <td>0.007542</td>\n      <td>0.234303</td>\n      <td>3.409430</td>\n      <td>7.656873</td>\n      <td>1788.390691</td>\n      <td>291.509045</td>\n      <td>1971.005136</td>\n      <td>84.402258</td>\n      <td>98077.939805</td>\n      <td>47.560053</td>\n      <td>-122.213896</td>\n      <td>1986.552492</td>\n      <td>12768.455652</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>3.671272e+05</td>\n      <td>0.926657</td>\n      <td>0.768996</td>\n      <td>918.440897</td>\n      <td>4.142051e+04</td>\n      <td>0.539989</td>\n      <td>0.086517</td>\n      <td>0.766318</td>\n      <td>0.650743</td>\n      <td>1.175459</td>\n      <td>828.090978</td>\n      <td>442.575043</td>\n      <td>29.373411</td>\n      <td>401.679240</td>\n      <td>53.505026</td>\n      <td>0.138564</td>\n      <td>0.140828</td>\n      <td>685.391304</td>\n      <td>27304.179631</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>7.500000e+04</td>\n      <td>1.000000</td>\n      <td>0.500000</td>\n      <td>290.000000</td>\n      <td>5.200000e+02</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>1.000000</td>\n      <td>1.000000</td>\n      <td>290.000000</td>\n      <td>0.000000</td>\n      <td>1900.000000</td>\n      <td>0.000000</td>\n      <td>98001.000000</td>\n      <td>47.155900</td>\n      <td>-122.519000</td>\n      <td>399.000000</td>\n      <td>651.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>3.219500e+05</td>\n      <td>3.000000</td>\n      <td>1.750000</td>\n      <td>1427.000000</td>\n      <td>5.040000e+03</td>\n      <td>1.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>3.000000</td>\n      <td>7.000000</td>\n      <td>1190.000000</td>\n      <td>0.000000</td>\n      <td>1951.000000</td>\n      <td>0.000000</td>\n      <td>98033.000000</td>\n      <td>47.471000</td>\n      <td>-122.328000</td>\n      <td>1490.000000</td>\n      <td>5100.000000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>4.500000e+05</td>\n      <td>3.000000</td>\n      <td>2.250000</td>\n      <td>1910.000000</td>\n      <td>7.618000e+03</td>\n      <td>1.500000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>3.000000</td>\n      <td>7.000000</td>\n      <td>1560.000000</td>\n      <td>0.000000</td>\n      <td>1975.000000</td>\n      <td>0.000000</td>\n      <td>98065.000000</td>\n      <td>47.571800</td>\n      <td>-122.230000</td>\n      <td>1840.000000</td>\n      <td>7620.000000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>6.450000e+05</td>\n      <td>4.000000</td>\n      <td>2.500000</td>\n      <td>2550.000000</td>\n      <td>1.068800e+04</td>\n      <td>2.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n      <td>4.000000</td>\n      <td>8.000000</td>\n      <td>2210.000000</td>\n      <td>560.000000</td>\n      <td>1997.000000</td>\n      <td>0.000000</td>\n      <td>98118.000000</td>\n      <td>47.678000</td>\n      <td>-122.125000</td>\n      <td>2360.000000</td>\n      <td>10083.000000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>7.700000e+06</td>\n      <td>33.000000</td>\n      <td>8.000000</td>\n      <td>13540.000000</td>\n      <td>1.651359e+06</td>\n      <td>3.500000</td>\n      <td>1.000000</td>\n      <td>4.000000</td>\n      <td>5.000000</td>\n      <td>13.000000</td>\n      <td>9410.000000</td>\n      <td>4820.000000</td>\n      <td>2015.000000</td>\n      <td>2015.000000</td>\n      <td>98199.000000</td>\n      <td>47.777600</td>\n      <td>-121.315000</td>\n      <td>6210.000000</td>\n      <td>871200.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
                        "text/plain": "              price      bedrooms     bathrooms   sqft_living      sqft_lot  \\\ncount  2.161300e+04  21600.000000  21603.000000  21613.000000  2.161300e+04   \nmean   5.400881e+05      3.372870      2.115736   2079.899736  1.510697e+04   \nstd    3.671272e+05      0.926657      0.768996    918.440897  4.142051e+04   \nmin    7.500000e+04      1.000000      0.500000    290.000000  5.200000e+02   \n25%    3.219500e+05      3.000000      1.750000   1427.000000  5.040000e+03   \n50%    4.500000e+05      3.000000      2.250000   1910.000000  7.618000e+03   \n75%    6.450000e+05      4.000000      2.500000   2550.000000  1.068800e+04   \nmax    7.700000e+06     33.000000      8.000000  13540.000000  1.651359e+06   \n\n             floors    waterfront          view     condition         grade  \\\ncount  21613.000000  21613.000000  21613.000000  21613.000000  21613.000000   \nmean       1.494309      0.007542      0.234303      3.409430      7.656873   \nstd        0.539989      0.086517      0.766318      0.650743      1.175459   \nmin        1.000000      0.000000      0.000000      1.000000      1.000000   \n25%        1.000000      0.000000      0.000000      3.000000      7.000000   \n50%        1.500000      0.000000      0.000000      3.000000      7.000000   \n75%        2.000000      0.000000      0.000000      4.000000      8.000000   \nmax        3.500000      1.000000      4.000000      5.000000     13.000000   \n\n         sqft_above  sqft_basement      yr_built  yr_renovated       zipcode  \\\ncount  21613.000000   21613.000000  21613.000000  21613.000000  21613.000000   \nmean    1788.390691     291.509045   1971.005136     84.402258  98077.939805   \nstd      828.090978     442.575043     29.373411    401.679240     53.505026   \nmin      290.000000       0.000000   1900.000000      0.000000  98001.000000   \n25%     1190.000000       0.000000   1951.000000      0.000000  98033.000000   \n50%     1560.000000       0.000000   1975.000000      0.000000  98065.000000   \n75%     2210.000000     560.000000   1997.000000      0.000000  98118.000000   \nmax     9410.000000    4820.000000   2015.000000   2015.000000  98199.000000   \n\n                lat          long  sqft_living15     sqft_lot15  \ncount  21613.000000  21613.000000   21613.000000   21613.000000  \nmean      47.560053   -122.213896    1986.552492   12768.455652  \nstd        0.138564      0.140828     685.391304   27304.179631  \nmin       47.155900   -122.519000     399.000000     651.000000  \n25%       47.471000   -122.328000    1490.000000    5100.000000  \n50%       47.571800   -122.230000    1840.000000    7620.000000  \n75%       47.678000   -122.125000    2360.000000   10083.000000  \nmax       47.777600   -121.315000    6210.000000  871200.000000  "
                    },
                    "execution_count": 10,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df.drop(['id','Unnamed: 0'],axis=1, inplace=True)\ndf.describe()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "We can see we have missing values for the columns <code> bedrooms</code>  and <code> bathrooms </code>"
        },
        {
            "cell_type": "code",
            "execution_count": 15,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "(21613, 20)"
                    },
                    "execution_count": 15,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df.shape"
        },
        {
            "cell_type": "code",
            "execution_count": 11,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": "number of NaN values for the column bedrooms : 13\nnumber of NaN values for the column bathrooms : 10\n"
                }
            ],
            "source": "print(\"number of NaN values for the column bedrooms :\", df['bedrooms'].isnull().sum())\nprint(\"number of NaN values for the column bathrooms :\", df['bathrooms'].isnull().sum())\n"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\nWe can replace the missing values of the column <code>'bedrooms'</code> with the mean of the column  <code>'bedrooms' </code> using the method <code>replace()</code>. Don't forget to set the <code>inplace</code> parameter to <code>True</code>"
        },
        {
            "cell_type": "code",
            "execution_count": 12,
            "metadata": {},
            "outputs": [],
            "source": "mean=df['bedrooms'].mean()\ndf['bedrooms'].replace(np.nan,mean, inplace=True)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\nWe also replace the missing values of the column <code>'bathrooms'</code> with the mean of the column  <code>'bathrooms' </code> using the method <code>replace()</code>. Don't forget to set the <code> inplace </code>  parameter top <code> True </code>"
        },
        {
            "cell_type": "code",
            "execution_count": 13,
            "metadata": {},
            "outputs": [],
            "source": "mean=df['bathrooms'].mean()\ndf['bathrooms'].replace(np.nan,mean, inplace=True)"
        },
        {
            "cell_type": "code",
            "execution_count": 14,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": "number of NaN values for the column bedrooms : 0\nnumber of NaN values for the column bathrooms : 0\n"
                }
            ],
            "source": "print(\"number of NaN values for the column bedrooms :\", df['bedrooms'].isnull().sum())\nprint(\"number of NaN values for the column bathrooms :\", df['bathrooms'].isnull().sum())"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Module 3: Exploratory Data Analysis"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 3\nUse the method <code>value_counts</code> to count the number of houses with unique floor values, use the method <code>.to_frame()</code> to convert it to a dataframe.\n"
        },
        {
            "cell_type": "code",
            "execution_count": 21,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>floors</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1.0</th>\n      <td>10680</td>\n    </tr>\n    <tr>\n      <th>2.0</th>\n      <td>8241</td>\n    </tr>\n    <tr>\n      <th>1.5</th>\n      <td>1910</td>\n    </tr>\n    <tr>\n      <th>3.0</th>\n      <td>613</td>\n    </tr>\n    <tr>\n      <th>2.5</th>\n      <td>161</td>\n    </tr>\n    <tr>\n      <th>3.5</th>\n      <td>8</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
                        "text/plain": "     floors\n1.0   10680\n2.0    8241\n1.5    1910\n3.0     613\n2.5     161\n3.5       8"
                    },
                    "execution_count": 21,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df['floors'].value_counts().to_frame()"
        },
        {
            "cell_type": "code",
            "execution_count": 25,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>waterfront</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>21450</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>163</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
                        "text/plain": "   waterfront\n0       21450\n1         163"
                    },
                    "execution_count": 25,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df['waterfront'].value_counts().to_frame()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 4\nUse the function <code>boxplot</code> in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers."
        },
        {
            "cell_type": "code",
            "execution_count": 33,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "<matplotlib.axes._subplots.AxesSubplot at 0x7fb143d5dba8>"
                    },
                    "execution_count": 33,
                    "metadata": {},
                    "output_type": "execute_result"
                },
                {
                    "data": {
                        "image/png": "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\n",
                        "text/plain": "<Figure size 432x288 with 1 Axes>"
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": "sns.set(style=\"whitegrid\")\nsns.boxplot(x=df['waterfront'],y=df['price'])"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 5\nUse the function <code>regplot</code>  in the seaborn library  to  determine if the feature <code>sqft_above</code> is negatively or positively correlated with price."
        },
        {
            "cell_type": "code",
            "execution_count": 34,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "<matplotlib.axes._subplots.AxesSubplot at 0x7fb143d426d8>"
                    },
                    "execution_count": 34,
                    "metadata": {},
                    "output_type": "execute_result"
                },
                {
                    "data": {
                        "image/png": "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\n",
                        "text/plain": "<Figure size 432x288 with 1 Axes>"
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": "sns.regplot(x=df['sqft_above'],y=df['price'],data=df)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\nWe can use the Pandas method <code>corr()</code>  to find the feature other than price that is most correlated with price."
        },
        {
            "cell_type": "code",
            "execution_count": 35,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "zipcode         -0.053203\nlong             0.021626\ncondition        0.036362\nyr_built         0.054012\nsqft_lot15       0.082447\nsqft_lot         0.089661\nyr_renovated     0.126434\nfloors           0.256794\nwaterfront       0.266369\nlat              0.307003\nbedrooms         0.308797\nsqft_basement    0.323816\nview             0.397293\nbathrooms        0.525738\nsqft_living15    0.585379\nsqft_above       0.605567\ngrade            0.667434\nsqft_living      0.702035\nprice            1.000000\nName: price, dtype: float64"
                    },
                    "execution_count": 35,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "df.corr()['price'].sort_values()"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Module 4: Model Development"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\nWe can Fit a linear regression model using the  longitude feature <code>'long'</code> and  caculate the R^2."
        },
        {
            "cell_type": "code",
            "execution_count": 36,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "0.00046769430149007363"
                    },
                    "execution_count": 36,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "X = df[['long']]\nY = df['price']\nlm = LinearRegression()\nlm.fit(X,Y)\nlm.score(X, Y)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question  6\nFit a linear regression model to predict the <code>'price'</code> using the feature <code>'sqft_living'</code> then calculate the R^2. Take a screenshot of your code and the value of the R^2."
        },
        {
            "cell_type": "code",
            "execution_count": 42,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "0.49285321790379316"
                    },
                    "execution_count": 42,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "X = df[['sqft_living']]\nY = df['price']\nlm1 = LinearRegression()\nlm1.fit(X,Y) \nlm1.score(X,Y)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 7\nFit a linear regression model to predict the <code>'price'</code> using the list of features:"
        },
        {
            "cell_type": "code",
            "execution_count": 43,
            "metadata": {},
            "outputs": [],
            "source": "features =[\"floors\", \"waterfront\",\"lat\" ,\"bedrooms\" ,\"sqft_basement\" ,\"view\" ,\"bathrooms\",\"sqft_living15\",\"sqft_above\",\"grade\",\"sqft_living\"]     "
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "Then calculate the R^2. Take a screenshot of your code."
        },
        {
            "cell_type": "code",
            "execution_count": 44,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "0.657679183672129"
                    },
                    "execution_count": 44,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "lm2 = LinearRegression()\nlm2.fit(df[features],Y)\nlm2.score(df[features],Y)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### This will help with Question 8\n\nCreate a list of tuples, the first element in the tuple contains the name of the estimator:\n\n<code>'scale'</code>\n\n<code>'polynomial'</code>\n\n<code>'model'</code>\n\nThe second element in the tuple  contains the model constructor \n\n<code>StandardScaler()</code>\n\n<code>PolynomialFeatures(include_bias=False)</code>\n\n<code>LinearRegression()</code>\n"
        },
        {
            "cell_type": "code",
            "execution_count": 45,
            "metadata": {},
            "outputs": [],
            "source": "Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 8\nUse the list to create a pipeline object to predict the 'price', fit the object using the features in the list <code>features</code>, and calculate the R^2."
        },
        {
            "cell_type": "code",
            "execution_count": 48,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "name": "stderr",
                    "output_type": "stream",
                    "text": "/opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/preprocessing/data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n  return self.partial_fit(X, y)\n/opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/base.py:467: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n  return self.fit(X, y, **fit_params).transform(X)\n/opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/pipeline.py:511: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n  Xt = transform.transform(Xt)\n"
                },
                {
                    "data": {
                        "text/plain": "0.7513408553309376"
                    },
                    "execution_count": 48,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "pipe = Pipeline(Input)\npipe.fit(df[features],Y)\npipe.score(df[features],Y)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Module 5: Model Evaluation and Refinement"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "Import the necessary modules:"
        },
        {
            "cell_type": "code",
            "execution_count": 49,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": "done\n"
                }
            ],
            "source": "from sklearn.model_selection import cross_val_score\nfrom sklearn.model_selection import train_test_split\nprint(\"done\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "We will split the data into training and testing sets:"
        },
        {
            "cell_type": "code",
            "execution_count": 50,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": "number of test samples: 3242\nnumber of training samples: 18371\n"
                }
            ],
            "source": "features =[\"floors\", \"waterfront\",\"lat\" ,\"bedrooms\" ,\"sqft_basement\" ,\"view\" ,\"bathrooms\",\"sqft_living15\",\"sqft_above\",\"grade\",\"sqft_living\"]    \nX = df[features]\nY = df['price']\n\nx_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)\n\n\nprint(\"number of test samples:\", x_test.shape[0])\nprint(\"number of training samples:\",x_train.shape[0])"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 9\nCreate and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data. \n"
        },
        {
            "cell_type": "code",
            "execution_count": 51,
            "metadata": {},
            "outputs": [],
            "source": "from sklearn.linear_model import Ridge"
        },
        {
            "cell_type": "code",
            "execution_count": 58,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "0.6478759163939121"
                    },
                    "execution_count": 58,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "ridge = Ridge(alpha=0.1)\nridge.fit(x_train,y_train)\nridge.score(x_test,y_test)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Question 10\nPerform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2."
        },
        {
            "cell_type": "code",
            "execution_count": 59,
            "metadata": {
                "jupyter": {
                    "outputs_hidden": false
                }
            },
            "outputs": [
                {
                    "data": {
                        "text/plain": "0.7002744279699229"
                    },
                    "execution_count": 59,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": "pr = PolynomialFeatures(degree=2)\nx_train_pr = pr.fit_transform(x_train)\nx_test_pr = pr.fit_transform(x_test)\n\nridge_pr = Ridge(alpha=0.1)\nridge_pr.fit(x_train_pr,y_train)\nridge_pr.score(x_test_pr,y_test)"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "<p>Once you complete your notebook you will have to share it. Select the icon on the top right a marked in red in the image below, a dialogue box should open, and select the option all&nbsp;content excluding sensitive code cells.</p>\n        <p><img width=\"600\" src=\"https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/save_notebook.png\" alt=\"share notebook\"  style=\"display: block; margin-left: auto; margin-right: auto;\"/></p>\n        <p></p>\n        <p>You can then share the notebook&nbsp; via a&nbsp; URL by scrolling down as shown in the following image:</p>\n        <p style=\"text-align: center;\"><img width=\"600\"  src=\"https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/url_notebook.png\" alt=\"HTML\" style=\"display: block; margin-left: auto; margin-right: auto;\" /></p>\n        <p>&nbsp;</p>"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "<h2>About the Authors:</h2> \n\n<a href=\"https://www.linkedin.com/in/joseph-s-50398b136/\">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD."
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "Other contributors: <a href=\"https://www.linkedin.com/in/michelleccarey/\">Michelle Carey</a>, <a href=\"www.linkedin.com/in/jiahui-mavis-zhou-a4537814a\">Mavis Zhou</a> "
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": ""
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.6",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.6.9"
        },
        "widgets": {
            "state": {},
            "version": "1.1.2"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}