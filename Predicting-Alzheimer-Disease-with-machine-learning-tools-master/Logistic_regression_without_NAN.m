function [results, tag_test_without_NAN] = Logistic_regression_without_NAN(set_without_NAN, set_tags_without_NAN)
    % Convert target tags to categorical
    t_ = categorical(set_tags_without_NAN);

    % Initialize train and test sets
    train_without_NAN = [];
    tag_train_without_NAN = [];
    test_without_NAN = [];
    tag_test_without_NAN = [];

    % Randomize indices for splitting
    num_samples = size(set_without_NAN, 1); % Get total number of samples
    idx = randperm(num_samples); % Randomize indices

    % Split into 70% training and 30% testing
    num_train = round(0.7 * num_samples); % Calculate training size
    train_without_NAN = set_without_NAN(idx(1:num_train), :);
    tag_train_without_NAN = t_(idx(1:num_train));

    test_without_NAN = set_without_NAN(idx(num_train+1:end), :);
    tag_test_without_NAN = t_(idx(num_train+1:end));

    % Train logistic regression model
    [B, dev] = mnrfit(train_without_NAN, tag_train_without_NAN);

    % Predict on test data
    results = mnrval(B, test_without_NAN);

    % Evaluate predictions
    correct_predictions = 0;
    for j = 1:size(test_without_NAN, 1)
        if (results(j, 1) > results(j, 2)) && (tag_test_without_NAN(j) == 'Demented')
            correct_predictions = correct_predictions + 1;
        elseif (results(j, 2) > results(j, 1)) && (tag_test_without_NAN(j) == 'Nondemented')
            correct_predictions = correct_predictions + 1;
        end
    end

    % Display results
    disp(['Data without NANs: The Logistic Regression predicts correctly ', ...
          num2str(correct_predictions), ' out of ', num2str(size(test_without_NAN, 1)), ' samples.']);
end
