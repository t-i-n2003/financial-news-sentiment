import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from processing.processing import get_score
#from processing.NLP import get_embedding, natural_language_process
from processing.NLP import natural_language_process
from sklearn.model_selection import GridSearchCV
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
def MLPClassifier_tune(X_train, y_train, model):
    param_grid = {
        'hidden_layer_sizes': [(256, 32), (100, 50), (50, 100, 50)],
        'activation': ['relu'],
        'solver': ['adam'],
        'batch_size': [64],
        'max_iter': [ 2000, 3000],
        'alpha': [0.01],
        'learning_rate': ['constant', 'adaptive']
    }

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best parameters found: ", grid_search.best_params_)
    pd.DataFrame([grid_search.best_params_]).to_csv('best_params_mlp.csv')
    print("Best cross-validation score: {:.2f}%".format(grid_search.best_score_ * 100))
    model_path = f'models/clf.pkl'
    joblib.dump(best_model, model_path)
    return best_model

def SVC_tune(X_train, y_train, model):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],
        'class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best parameters found: ", grid_search.best_params_)
    pd.DataFrame([grid_search.best_params_]).to_csv('best_params_svc.csv')
    print("Best cross-validation score: {:.2f}%".format(grid_search.best_score_ * 100))
    model_path = f'models/svc.pkl'
    joblib.dump(best_model, model_path)
    return best_model

def map_score(score):
    if score < -0.3:
        return 0 
    elif score > 0.3:
        return 2 
    else:
        return 1

# data = get_score(['ACB', 'BID', 'VCB', 'MBB'], file = 'dataset/news.csv', start_date ='2021-01-01', end_date ='2025-01-01')
# data['label'] = data['score'].apply(map_score)
# data['title'] = data['title'].apply(lambda x: natural_language_process(x, symbols = ['ACB', 'BID', 'VCB', 'MBB']))
# data.to_excel('dataset/processed_data.xlsx', index=False)


data = pd.read_csv(f'dataset/processed_data.csv')
# X = data['title'].apply(get_embedding).tolist()
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['title']).toarray()
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
# input_dim = X.shape[1]
# model = nn.Sequential(
#     nn.Linear(input_dim, 128), 
#     nn.ReLU(),
#     nn.Linear(128, 3),
#     nn.Softmax(dim=1)
# )
X_train, X_test, y_train, y_test = train_test_split(X, data['label'].values, test_size=0.2, random_state=50, stratify=data['label'].values)

# train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                               torch.tensor(y_train, dtype=torch.long))
# test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
#                              torch.tensor(y_test, dtype=torch.long))

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()
# # Train loop
# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * batch_X.size(0)

#     epoch_loss = running_loss / len(train_loader.dataset)

#     model.eval()
#     with torch.no_grad():
#         # Accuracy train
#         correct_train = 0
#         total_train = 0
#         for batch_X, batch_y in train_loader:
#             outputs = model(batch_X)
#             _, predicted = torch.max(outputs, 1)
#             total_train += batch_y.size(0)
#             correct_train += (predicted == batch_y).sum().item()
#         train_acc = 100 * correct_train / total_train

#         correct_test = 0
#         total_test = 0
#         for batch_X, batch_y in test_loader:
#             outputs = model(batch_X)
#             _, predicted = torch.max(outputs, 1)
#             total_test += batch_y.size(0)
#             correct_test += (predicted == batch_y).sum().item()
#         test_acc = 100 * correct_test / total_test

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")




# X = np.array(X)
# X = np.array([np.array(x) for x in X])
# Y = data['label'].values
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state= 50, stratify=Y)

# mlp = MLPClassifier(
#     hidden_layer_sizes=(100, 50), 
#     activation='relu', 
#     solver='adam', 
#     batch_size=64, 
#     max_iter=2000, 
#     alpha=0.01, 
#     learning_rate='adaptive'
# )
# model = mlp.fit(X_train, y_train)
# # model = MLPClassifier_tune(X_train, y_train, mlp)

svc = SVC(
    kernel='rbf', 
    C=1, 
    gamma='scale',
    degree=2,
    class_weight= None
)
model = svc.fit(X_train, y_train)
# model = SVC_tune(X_train, y_train, svc)
# joblib.dump(model, 'models/svc.pkl')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
