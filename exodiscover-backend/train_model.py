import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, class_weight
from sklearn.metrics import classification_report
import pickle

# --- Load CSV ---
df = pd.read_csv(r"model\lighkurve_KOI_dataset.csv")

# --- Features and labels ---
feature_cols = ['koi_period', 'koi_period_err1', 'koi_period_err2',
                'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
                'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
                'koi_quarters']

# Ensure features are numeric
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df['koi_disposition']

# Fill missing values
X = X.fillna(X.mean())

# Encode labels
le = LabelEncoder()
y_int = le.fit_transform(y)

# --- Combine features and labels for oversampling ---
df_encoded = X.copy()
df_encoded['label'] = y_int

# --- Stratified train/test split (70/30) ---
df_train, df_test = train_test_split(df_encoded, test_size=0.3, random_state=42, stratify=y_int)

# --- Oversample training set to balance classes ---
max_count = df_train['label'].value_counts().max()
balanced_train = []

for label in df_train['label'].unique():
    subset = df_train[df_train['label'] == label]
    oversampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_train.append(oversampled)

df_train_balanced = pd.concat(balanced_train)

X_train = df_train_balanced[feature_cols].values
y_train = to_categorical(df_train_balanced['label'].values)

X_test = df_test[feature_cols].values
y_test = to_categorical(df_test['label'].values)

# --- Scale features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Compute class weights ---
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(weights))

# --- Build model ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train model with early stopping ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.3f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

# --- Save model, scaler, and label encoder ---
model.save("lightcurve_model.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModel, scaler, and label encoder saved successfully!")
