
import numpy as np
import pandas as pd

def create_features(df):
    # Calculates log-transformed Body Mass Index (BMI) for better normalization of data.
    df["log_BMI"] = np.log(df["Weight"] / (df["Height"] ** 2) + 1e-6)

    # Categorizes individuals into age groups to identify age-related trends.
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 40, 50, 60, np.inf],
        labels=["<18", "18-30", "30-40", "40-50", "50-60", "60+"],
    )

    # Computes a physical activity score to quantify participants' physical activity levels.
    df["physical_activity_score"] = df["FAF"] * 2 + (1 - df["TUE"] / 2)

    # Determines the tendency of caloric intake based on frequency of consuming high-caloric food and vegetables.
    df["caloric_intake_tendency"] = (
        df["FAVC"].apply(lambda x: 1 if x == "yes" else 0)
        + (df["FCVC"] / 3)
        + (1 - (df["NCP"] / 4))
    )
    
    # Mapping CAEC values to numeric scores.
    caec_mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df["CAEC"] = df["CAEC"].map(caec_mapping)

    # Establishes a healthy eating score based on vegetable consumption frequency and eating habits.
    df["healthy_eating_score"] = (
        2 * df["FCVC"]
        + df["NCP"]
        - df["CAEC"]
    )

    # Calculates Basal Metabolic Rate (BMR) for each individual using the Mifflin-St Jeor Equation, considering gender differentiation.
    df["BMR"] = df.apply(
        lambda row: (10 * row["Weight"])
        + (6.25 * (row["Height"] * 100))
        - (5 * row["Age"])
        + (5 if row["Gender"] == "Male" else -161),
        axis=1,
    )

    # Evaluates meal regularity by counting the main meals.
    df["meal_regularity_score"] = df["NCP"].apply(lambda x: 1 if x >= 3 else 0)

    # Assesses snacking habit based on consumption of high caloric food and sweets.
    df["snacking_habit"] = df["FAVC"].apply(lambda x: 1 if x == "yes" else 0) + df[
        "CAEC"
    ]

    # Indicates stress eating based on the relationship of physical activity score and meal regularity.
    df["stress_eating_indicator"] = (2 - df["physical_activity_score"]) * (
        1 - df["meal_regularity_score"] / 2
    )

    # Scores sedentary lifestyle based on time spent using technology devices.
    df["sedentary_lifestyle_score"] = df["TUE"].apply(
        lambda x: 2 if x > 4 else 1 if x <= 4 and x > 0 else 0
    )

    # Calculates an overall lifestyle score combining physical activity, healthy eating, meal regularity and less snacking habit.
    df["overall_lifestyle_score"] = (
        df["physical_activity_score"]
        + df["healthy_eating_score"]
        + df["meal_regularity_score"]
        - df["snacking_habit"]
        + df["stress_eating_indicator"] * 2 
        + df["sedentary_lifestyle_score"] * 2 
        + 1
    )
    
    # Feature generation (exp, log, sqrt, interact, etc.)
    # Log transformation for continuous variables
    df["log_Age"] = np.log(df["Age"] + 1)
    df["log_Height"] = np.log(df["Height"] + 1)
    df["log_Weight"] = np.log(df["Weight"] + 1)

    # Square root transformation
    df["sqrt_FCVC"] = np.sqrt(df["FCVC"])
    df["sqrt_NCP"] = np.sqrt(df["NCP"])
    df["sqrt_CH2O"] = np.sqrt(df["CH2O"])

    # Interaction terms
    df["Height_Weight_interaction"] = df["Height"] * df["Weight"]
    df["Age_FCVC_interaction"] = df["Age"] * df["FCVC"]

    return df
    