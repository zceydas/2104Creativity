%% AUT Flexibility Calculator
% Reads a coded AUT CSV and computes:
% 1) Flexibility per Subject x Session x Dose x Item  (# unique categories)
% 2) Average flexibility per Subject x Session x Dose (mean across items)

clear; clc;

% --------- INPUTS ----------
inFile  = "aut_all_items_coded.csv";  % <-- change to your output filename
outItem = "flexibility_by_item.csv";
outAvg  = "flexibility_avg_by_subject_session_dose.csv";

% --------- READ ----------
T = readtable(inFile, 'TextType','string');

% --------- NORMALIZE COLUMN NAMES ----------
% Your headers include spaces. Use exact names as they appear.
subCol  = "Subject";
sessCol = "Session";
doseCol = "Dose";
itemCol = "item";
catCol  = "category";

required = [subCol sessCol doseCol itemCol catCol];
missing = required(~ismember(required, string(T.Properties.VariableNames)));
if ~isempty(missing)
    error("Missing required columns: %s", strjoin(missing, ", "));
end

% --------- FORCE STRING TYPE SAFELY ----------
T.(subCol)  = string(T.(subCol));
T.(sessCol) = string(T.(sessCol));
T.(doseCol) = string(T.(doseCol));
T.(itemCol) = string(T.(itemCol));
T.(catCol)  = string(T.(catCol));

% --------- CLEAN VALUES ----------
T.(subCol)  = strtrim(T.(subCol));
T.(sessCol) = strtrim(T.(sessCol));
T.(doseCol) = strtrim(T.(doseCol));
T.(itemCol) = lower(strtrim(T.(itemCol)));
T.(catCol)  = strtrim(T.(catCol));

% Optional: enforce dose levels (keeps ordering consistent)
doseLevels = ["Baseline","Placebo","10mg/70kg"];
T.(doseCol) = categorical(T.(doseCol), doseLevels, 'Ordinal', true);

% Drop rows with missing category (if any)
T = T(~ismissing(T.(catCol)) & T.(catCol) ~= "", :);

% --------- FLEXIBILITY PER SUBJECT x SESSION x DOSE x ITEM ----------
% Flexibility = number of distinct categories within that cell
[G, subj, sess, dose, item] = findgroups(T.(subCol), T.(sessCol), T.(doseCol), T.(itemCol));

flex = splitapply(@(c) numel(unique(c)), T.(catCol), G);

FlexByItem = table(subj, sess, dose, item, flex, ...
    'VariableNames', ["SubjectID","Session","Dose","Item","Flexibility"]);

% --------- AVERAGE FLEXIBILITY PER SUBJECT x SESSION x DOSE ----------
% Mean flexibility across items (within Subject x Session x Dose)
[G2, subj2, sess2, dose2] = findgroups(FlexByItem.SubjectID, FlexByItem.Session, FlexByItem.Dose);
avgFlex = splitapply(@mean, FlexByItem.Flexibility, G2);

AvgFlex = table(subj2, sess2, dose2, avgFlex, ...
    'VariableNames', ["SubjectID","Session","Dose","AvgFlexibility"]);

% --------- WRITE OUTPUTS ----------
writetable(FlexByItem, outItem);
writetable(AvgFlex, outAvg);

disp("Saved: " + outItem);
disp("Saved: " + outAvg);


%% Average flexibility per Subject × Session

% Ensure types are consistent
T.SubjectID = string(T.SubjectID);
T.Session   = string(T.Session);

% Group by Subject and Session
[G, subj, sess] = findgroups(T.SubjectID, T.Session);

avgFlex = splitapply(@mean, T.Flexibility, G);

AvgBySubSess = table(subj, sess, avgFlex, ...
    'VariableNames', ["SubjectID","Session","AvgFlexibility"]);

% Save if desired
writetable(AvgBySubSess, "avg_flexibility_by_subject_session.csv");

disp(AvgBySubSess)


%% clear; clc;
clear; clc;

inFile = "flexibility_avg_by_subject_session_dose.csv";  % <-- your file
T = readtable(inFile, 'TextType','string');

% Force types
T.SubjectID      = string(T.SubjectID);
T.Session        = double(T.Session);
T.Dose           = strtrim(string(T.Dose));
T.AvgFlexibility = double(T.AvgFlexibility);

% Dose order
doseLevels = ["Baseline","Placebo","10mg/70kg"];
T.Dose = categorical(T.Dose, doseLevels, 'Ordinal', true);

% Group id per row
G = findgroups(T.Dose);

% Stats per group (length = number of dose levels present)
meanFlex = splitapply(@mean, T.AvgFlexibility, G);
semFlex  = splitapply(@(x) std(x)/sqrt(numel(x)), T.AvgFlexibility, G);

% Get dose label per group in the same order as splitapply output
dosePerGroup = splitapply(@(x) x(1), double(T.Dose), G);  % numeric codes per group
dosePerGroup = categorical(dosePerGroup, 1:numel(doseLevels), doseLevels, 'Ordinal', true);

DoseSummary = table(dosePerGroup, meanFlex, semFlex, ...
    'VariableNames', {'Dose','MeanFlexibility','SEM'});

disp(DoseSummary);

% Plot
figure;
errorbar(DoseSummary.Dose, DoseSummary.MeanFlexibility, DoseSummary.SEM, 'o-', 'LineWidth', 2);
xlabel('Dose');
ylabel('Average Flexibility (across participants)');
title('AUT Flexibility by Dose (Mean \pm SEM)');
grid on;

writetable(DoseSummary, "dose_mean_sem_from_avgfile.csv");