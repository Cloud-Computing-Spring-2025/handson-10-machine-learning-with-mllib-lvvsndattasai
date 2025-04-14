from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load CSV
df = spark.read.csv("customer_churn.csv", header=True, inferSchema=True)

# Task 1: Preprocessing
def preprocess_data(df):
    df = df.na.fill({"TotalCharges": 0})

    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=col+"_indexed")
        df = indexer.fit(df).transform(df)
        
        encoder = OneHotEncoder(inputCol=col+"_indexed", outputCol=col+"_encoded")
        df = encoder.fit(df).transform(df)

    encoded_cols = [col+"_encoded" for col in categorical_cols]
    assembler = VectorAssembler(inputCols=numeric_cols + encoded_cols, outputCol="features")
    df = assembler.transform(df)

    df = StringIndexer(inputCol="Churn", outputCol="label").fit(df).transform(df)
    return df.select("features", "label")

# Task 2: Logistic Regression
def train_logistic_regression_model(df):
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression()
    model = lr.fit(train)
    predictions = model.transform(test)
    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    print(f"Logistic Regression AUC: {auc:.4f}")

# Task 3: Chi-Square
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    result = selector.fit(df).transform(df)
    result.select("selectedFeatures", "label").show(5, truncate=False)
    return result

# Task 4: Model Comparison with CrossValidator
def tune_and_compare_models(df):
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator()
    
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBT": GBTClassifier()
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [3, 5]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 20]).build(),
        "GBT": ParamGridBuilder().addGrid(models["GBT"].maxIter, [10, 20]).build()
    }

    for name in models:
        print(f"\n--- Training {name} ---")
        cv = CrossValidator(estimator=models[name],
                            estimatorParamMaps=param_grids[name],
                            evaluator=evaluator,
                            numFolds=5)

        cv_model = cv.fit(train)
        predictions = cv_model.transform(test)
        auc = evaluator.evaluate(predictions)
        print(f"{name} AUC: {auc:.4f}")

# Run All Tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark
spark.stop()

