import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def model_evaluation(model, X_test, y_test, results_dir, name=None):
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # print("Best Parameters:", grid.best_params_)
    # print(f"Y_pred values: {y_pred}")
    print(f"{name} R²: {r2:.4f}")
    print(f"{name} MAE: {mae:.4f}")
    print(f"{name} MSE: {mse:.4f}")
    print(f"{name} RMSE: {rmse:.4f}")


    ### Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(results_dir +  f"{name}" + "_residual_plot.png", dpi=400)
    plt.show()

    ### Histogram of residual plots
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residuals Distribution Plot")
    plt.tight_layout()
    plt.savefig(results_dir +  f"{name}" + "_histogram_residual_plot.png", dpi=400)
    plt.show()


    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)

    # Add the ideal diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Ideal Predictions')

    plt.xlabel("Actual Values ($y_{true}$)")
    plt.ylabel("Predicted Values ($y_{pred}$)")
    plt.title("Actual vs. Predicted Values Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir +  f"{name}" + "_scatter_plot_.png", dpi=400)
    plt.show()

    # Save best model 
    joblib.dump(model, results_dir +  f"{name}" + "_best_model.pkl")
    print(f"\{name} model saved to best_{name}_model.pkl")


