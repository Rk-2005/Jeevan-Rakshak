# JeevanRakshak

**JeevanRakshak** is a project that uses Flask for web development, EPANet for water network simulations, and various data analysis and machine learning tools to provide insights and predictions based on water network data.

## Setup the Project

To get started, download and extract the project files to your local machine.

## Setup and Installation

1. **Navigate to the project directory:**

    ```
    cd JeevanRakshak
    ```

2. **Create and activate a virtual environment:**

    ```
    python -m venv my_env
    source my_env/bin/activate  # On Windows use `my_env\Scripts\activate`
    ```

3. **Install the required packages:**

    ```
    pip install Flask Flask-Cors pandas numpy matplotlib seaborn scikit-learn tensorflow epanettools
    ```

## Project Requirements

This project requires the following Python packages:

- `Flask`
- `Flask-Cors`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `epanettools`

## Running the Flask Modules

To run the Python modules, use the following commands:

```
py waterquality.py
```
```
py sensor_allocation_final.py
```
```
py finalpilferage.py
```
```
py leak_model.py
```

# Running the React Applications

## Running GIS Tracking

1. **Navigate to the `gisTracking-main` directory:**

    ```
    cd gisTracking-main
    ```

2. **Install dependencies:**

    ```
    npm i
    ```

3. **Navigate to the `src` directory and start the application:**

    ```
    cd src
    npm start
    ```

4. **Navigate back to the root directory:**

    ```
    cd ..
    cd ..
    ```

## Running Dashboard

1. **Navigate to the `Dashboard` directory:**

    ```
    cd Dashboard
    ```

2. **Install dependencies:**

    ```
    npm i
    ```

3. **Navigate to the `src` directory and start the application:**

    ```
    cd src
    npm start
    ```

4. **Navigate back to the root directory:**

    ```
    cd ..
    ```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **EPANettools** for water network simulations.
- **TensorFlow and Keras** for machine learning models.
- **Flask** for creating the web application.
- **Various data analysis libraries** like `pandas`, `numpy`, and `matplotlib`.

