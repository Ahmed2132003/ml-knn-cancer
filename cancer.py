import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# تحميل البيانات من ملف CSV
df = pd.read_csv('D:\\programming project\\python\\ml\\knn\\knn_class\\KNNAlgorithmDataset.csv')

# معالجة القيم النصية في عمود 'diagnosis'
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# إزالة الأعمدة التي تحتوي على جميع القيم المفقودة
df = df.dropna(axis=1, how='all')

# تحويل القيم النصية إلى NaN
for col in df.select_dtypes(include=[object]).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# التعامل مع القيم المفقودة باستخدام SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# إعداد البيانات وتدريب نموذج KNN
X = df.drop(['diagnosis', 'id'], axis=1, errors='ignore')
y = df['diagnosis']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# مقياس البيانات
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تدريب نموذج KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# اختبار النموذج وحساب الدقة
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# إنشاء تطبيق Tkinter
root = tk.Tk()
root.title("KNN Cancer Prediction Analysis")
root.geometry("1200x1000")
root.configure(bg='#FFD700')  # تعيين الخلفية إلى اللون الذهبي
root.state("zoomed")
# إطار يحتوي على Canvas و Scrollbar
main_frame = tk.Frame(root, bg='#FFD700')
main_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(main_frame, bg='#FFD700')
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg='#FFD700')

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# تحليل الوصف
label = ttk.Label(scrollable_frame, text="1. Descriptive Analysis", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()
text = tk.Text(scrollable_frame, height=10, width=175, bg='#FFD700', fg='white')
text.pack(fill="x")
text.insert(tk.END, df.describe().to_string())

# رسم مصفوفة الترابط
label = ttk.Label(scrollable_frame, text="2. Correlation Matrix", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FFD700')  # تغيير لون خلفية الرسم البياني
ax.set_facecolor('#FFD700')         # تغيير لون خلفية المحور
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
canvas_corr = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_corr.draw()
canvas_corr.get_tk_widget().pack()

# رسم توزيع البيانات
label = ttk.Label(scrollable_frame, text="3. Data Distribution (Radius Mean)", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()
fig, ax = plt.subplots()
fig.patch.set_facecolor('#FFD700')  # تغيير لون خلفية الرسم البياني
ax.set_facecolor('#FFD700')         # تغيير لون خلفية المحور
sns.histplot(df['radius_mean'], bins=20, ax=ax, color='skyblue')
canvas_dist = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_dist.draw()
canvas_dist.get_tk_widget().pack()

# رسم زوجي لتحليل العلاقات بين المتغيرات
label = ttk.Label(scrollable_frame, text="4. Pairplot for Feature Relationships", background='#FFD700', foreground='white', font=('tajawal', '16', 'bold'))
label.pack()

# إنشاء الرسم البياني
fig = sns.pairplot(df[['radius_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']])

# تغيير لون خلفية جميع المحاور في الرسم البياني إلى الذهبي
for ax in fig.axes.flatten():
    ax.set_facecolor('#FFD700')  # تغيير لون خلفية كل محور

# تغيير لون خلفية الشكل الأساسي للرسم
fig.fig.patch.set_facecolor('#FFD700')

# عرض الرسم البياني في تطبيق Tkinter
fig_canvas = FigureCanvasTkAgg(fig.fig, master=scrollable_frame)
fig_canvas.draw()
fig_canvas.get_tk_widget().pack()

# تحليل القيم المتطرفة باستخدام Boxplot
label = ttk.Label(scrollable_frame, text="5. Outlier Analysis", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()
fig, ax = plt.subplots()
fig.patch.set_facecolor('#FFD700')  # تغيير لون خلفية الرسم البياني
ax.set_facecolor('#FFD700')         # تغيير لون خلفية المحور
sns.boxplot(x=df['texture_mean'], ax=ax, color='lightgreen')
canvas_outliers = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_outliers.draw()
canvas_outliers.get_tk_widget().pack()

# عرض دقة النموذج
label = ttk.Label(scrollable_frame, text=f"6. Model Accuracy: {accuracy * 100:.2f}%", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()

# رسم مصفوفة التشوش
label = ttk.Label(scrollable_frame, text="7. Confusion Matrix", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
fig.patch.set_facecolor('#FFD700')  # تغيير لون خلفية الرسم البياني
ax.set_facecolor('#FFD700')         # تغيير لون خلفية المحور
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', ax=ax)
canvas_conf_matrix = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_conf_matrix.draw()
canvas_conf_matrix.get_tk_widget().pack()

# عرض تقرير التصنيف
label = ttk.Label(scrollable_frame, text="8. Classification Report", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()
text = tk.Text(scrollable_frame, height=10, width=100, bg='#FFD700', fg='white')
text.pack()
text.insert(tk.END, classification_report(y_test, y_pred))

# التنبؤ بناءً على إدخال المستخدم
label = ttk.Label(scrollable_frame, text="9. Enter values to make a prediction", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
label.pack()

# إعداد إدخالات المستخدم - وضع 5 حقول إدخال في كل صف
features = list(X.columns)  # جميع الميزات
inputs = {}

for i in range(0, len(features), 5):
    row_frame = tk.Frame(scrollable_frame, bg='#FFD700')
    row_frame.pack(fill=tk.X, padx=10, pady=5)
    for feature in features[i:i+5]:
        label = ttk.Label(row_frame, text=f"{feature.replace('_', ' ')}", background='#FFD700', foreground='white')
        label.pack(side=tk.LEFT, padx=5, pady=5)
        entry = ttk.Entry(row_frame)
        entry.pack(side=tk.LEFT, padx=5, pady=5)
        inputs[feature] = entry

# زر التنبؤ
def predict():
    try:
        user_data = [[float(inputs[feature].get()) for feature in inputs]]
        user_data = scaler.transform(user_data)
        prediction = knn.predict(user_data)
        result = "Benign" if prediction[0] == 0 else "Malignant"
        result_label.config(text=f"Result: {result}")
    except ValueError:
        result_label.config(text="Error: Please enter valid numbers.")

predict_button = ttk.Button(scrollable_frame, text="Predict", command=predict)
predict_button.pack(pady=10)

result_label = ttk.Label(scrollable_frame, text="Result: ", background='#FFD700', foreground='white',font=('tajawal','16','bold'))
result_label.pack()

root.mainloop()
