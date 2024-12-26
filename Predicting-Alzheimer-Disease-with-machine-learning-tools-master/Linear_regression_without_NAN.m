% Load the dataset
data = readtable('alzheimers_disease_data.csv');

% Drop irrelevant columns
if ismember('PatientID', data.Properties.VariableNames)
    data.PatientID = [];
end
if ismember('DoctorInCharge', data.Properties.VariableNames)
    data.DoctorInCharge = [];
end

% Identify categorical columns
categoricalCols = varfun(@iscategorical, data, 'OutputFormat', 'uniform');

% Convert categorical columns to numeric using grp2idx
for col = find(categoricalCols)
    columnName = data.Properties.VariableNames{col};
    data.(columnName) = grp2idx(data.(columnName));
end

% Ensure the target column (Diagnosis) is numeric
if iscategorical(data.Diagnosis)
    data.Diagnosis = grp2idx(data.Diagnosis);
end

% Split features and target
X = data(:, 1:end-1); % All columns except the last
y = data(:, end);     % Target column

% Convert tables to arrays
X = table2array(X);
y = table2array(y);

% Split into training and test sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% train, 20% test
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Train Gradient Boosting Model
rng('default'); % For reproducibility
GradBoostModel = fitcensemble(X_train, y_train, ...
    'Method', 'LogitBoost', 'NumLearningCycles', 100, 'Learners', 'Tree');

% Predict on test set
y_pred = predict(GradBoostModel, X_test);

% Evaluate the model
confMat = confusionmat(y_test, y_pred);
accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;

% Display results
disp('Confusion Matrix:');
disp(confMat);
disp(['Accuracy: ', num2str(accuracy), '%']);
