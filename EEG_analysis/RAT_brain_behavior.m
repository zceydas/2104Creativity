% =====================================================================
% RAT_brain_behavior.m
% ---------------------------------------------------------------------
% Correlate the psilocybin EEG band-power effect with the psilocybin
% behavioral effect, across subjects.
%
% Both sides are expressed as a drug effect referenced to each subject's
% own BASELINE (ses-0 / Session 1):
%     EEG_effect  = bandpower(psilocybin) - bandpower(baseline)
%     beh_effect  = score(psilocybin)     - score(baseline)
% (Placebo-referenced drug-placebo contrast also computed as a secondary.)
%
% PRE-SPECIFIED primary test:
%     alpha drug-minus-baseline  vs  RAT_Accuracy drug-minus-baseline
% Everything else (theta; AUT measures; drug-minus-placebo) is exploratory.
%
% Inputs:
%   EEG   : bandpower_long.csv  (subject x session x condition x band x channel)
%           -> we need per-subject-per-condition band power, so we also need
%              which EEG session was baseline. Baseline = ses-0 always.
%   Behav : EBRWMC_Updated.csv  (raw per-condition scores, incl. Baseline)
%
% Frequentist: Spearman (primary) + Pearson, per-test n, scatter, leave-one-out.
% Bayesian:    JZS default correlation Bayes factor (uniform prior on rho,
%              kappa = 1 -- the JASP default), computed on the Pearson r.
%              BF10 > 1 favours a correlation; BF10 < 1 (BF01 > 1) favours
%              the null. VERIFY the primary BF against JASP before publishing.
% =====================================================================

clearvars;

% ---- paths (local Mac layout) ----
ANALYSIS_DIR = '/Users/zsayali1/Documents/2104_Creativity/analysis';
EEG_LONG   = fullfile(ANALYSIS_DIR, 'bandpower_long.csv');
BEHAV_CSV  = '/Users/zsayali1/Documents/2104_Creativity/Data/Raw/EBRWMC_Updated.csv';
OUT_DIR    = fullfile(ANALYSIS_DIR, 'brain_behavior');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

% ---- pre-specification ----
PRIMARY_BAND = "alpha";
PRIMARY_BEH  = "RAT_Accuracy";
SEC_BANDS    = ["theta","alpha"];
SEC_BEH      = ["RAT_Accuracy","AUT_Originality","AUT_Fluency","AUT_Flexibility"];
DRUG_LABEL   = "10mg/70kg";      % Dose string for psilocybin in the behav file

% =====================================================================
% 1. EEG: per-subject band power by CONDITION, then baseline-reference
% =====================================================================
E = readtable(EEG_LONG, 'TextType','string');
% E columns: subject, session, condition, band, channel, power, n_trials
% subject like "sub-21040005" -> numeric 5
E.subjNum = arrayfun(@(s) mod(str2double(regexp(s,'\d+','match','once')),10000), E.subject);

% channel-average within subject x condition x band
Eg = groupsummary(E, {'subjNum','condition','band'}, 'mean', 'power');
Eg.Properties.VariableNames{'mean_power'} = 'power';
% condition here is "baseline"/"placebo"/"drug" (from the analysis script mapping)

eeg_effect = eeg_condition_diff(Eg, "drug", "baseline");    % drug - baseline
eeg_pla    = eeg_condition_diff(Eg, "drug", "placebo");     % drug - placebo (secondary)

% =====================================================================
% 2. Behaviour: raw per-condition -> drug-minus-baseline per subject
% =====================================================================
B = readtable(BEHAV_CSV, 'TextType','string');
B.Properties.VariableNames = matlab.lang.makeValidName(B.Properties.VariableNames);
pidName = B.Properties.VariableNames{find(contains(lower(B.Properties.VariableNames),'partic'),1)};
B.subjNum = double(B.(pidName));
% normalise dose
dcol = B.Properties.VariableNames{find(contains(lower(B.Properties.VariableNames),'dose'),1)};
dose = string(B.(dcol));
condB = strings(size(dose));
condB(contains(lower(dose),'placebo'))  = "placebo";
condB(contains(lower(dose),'baseline')) = "baseline";
condB(dose==DRUG_LABEL | contains(dose,'mg')) = "drug";
B.condB = condB;

behEffect = struct();   % behEffect.(measure) -> table subjNum, effect (drug-baseline)
for m = unique([PRIMARY_BEH SEC_BEH])
    behEffect.(m) = behav_condition_diff(B, m, "drug", "baseline");
end

% =====================================================================
% 3. PRIMARY correlation: alpha(drug-baseline) vs RAT_Accuracy(drug-baseline)
% =====================================================================
fprintf('\n===== PRIMARY (pre-specified) =====\n');
[rp, pp, np, bf10p] = run_corr(eeg_effect, PRIMARY_BAND, behEffect.(PRIMARY_BEH), ...
    sprintf('%s power (drug-baseline)',PRIMARY_BAND), ...
    sprintf('%s (drug-baseline)',PRIMARY_BEH), ...
    fullfile(OUT_DIR, sprintf('PRIMARY_%s_vs_%s.png',PRIMARY_BAND,PRIMARY_BEH)), true);
fprintf('  PRIMARY Bayes factor: BF10 = %.3f  |  BF01 = %.2f  (%s)\n', ...
        bf10p, 1/bf10p, bf_label(bf10p));

% =====================================================================
% 4. EXPLORATORY grid: {theta,alpha} x {RAT, AUT measures}
% =====================================================================
fprintf('\n===== EXPLORATORY (secondary) =====\n');
rows = {};
for bd = SEC_BANDS
    for be = SEC_BEH
        [r,p,n,bf] = run_corr(eeg_effect, bd, behEffect.(be), ...
            sprintf('%s (drug-baseline)',bd), sprintf('%s (drug-baseline)',be), ...
            fullfile(OUT_DIR, ['sec_' char(bd) '_vs_' char(be) '.png']), false);
        rows(end+1,:) = {bd, be, "drug-baseline", n, r, p, bf, 1/bf}; %#ok<AGROW>
    end
end
% secondary contrast: drug - placebo (baseline-invariant), same behaviour differenced d-pla
behEffectPla = struct();
for m = unique([PRIMARY_BEH SEC_BEH])
    behEffectPla.(m) = behav_condition_diff(B, m, "drug", "placebo");
end
for bd = SEC_BANDS
    for be = SEC_BEH
        [r,p,n,bf] = run_corr(eeg_pla, bd, behEffectPla.(be), ...
            sprintf('%s (drug-placebo)',bd), sprintf('%s (drug-placebo)',be), ...
            fullfile(OUT_DIR, ['secPLA_' char(bd) '_vs_' char(be) '.png']), false);
        rows(end+1,:) = {bd, be, "drug-placebo", n, r, p, bf, 1/bf}; %#ok<AGROW>
    end
end
Sgrid = cell2table(rows, 'VariableNames', ...
    {'band','behaviour','contrast','n','rho','p','BF10','BF01'});
writetable(Sgrid, fullfile(OUT_DIR,'exploratory_grid.csv'));
fprintf('\nExploratory grid (BF10>1 favours correlation; BF01>1 favours null):\n'); disp(Sgrid);
fprintf('\nResults in: %s\n', OUT_DIR);


% =====================================================================
% local functions
% =====================================================================
function T = eeg_condition_diff(Eg, cA, cB)
% per-subject (cA - cB) band power, wide over bands -> table subjNum + one col per band
bands = unique(Eg.band,'stable');
subs  = unique(Eg.subjNum);
T = table(subs,'VariableNames',{'subjNum'});
for b = 1:numel(bands)
    bn = bands(b); v = nan(numel(subs),1);
    for s = 1:numel(subs)
        pa = Eg.power(Eg.subjNum==subs(s) & Eg.condition==cA & Eg.band==bn);
        pb = Eg.power(Eg.subjNum==subs(s) & Eg.condition==cB & Eg.band==bn);
        if ~isempty(pa) && ~isempty(pb), v(s) = pa(1)-pb(1); end
    end
    T.(bn) = v;
end
end

function T = behav_condition_diff(B, measure, cA, cB)
subs = unique(B.subjNum);
v = nan(numel(subs),1);
for s = 1:numel(subs)
    xa = B.(measure)(B.subjNum==subs(s) & B.condB==cA);
    xb = B.(measure)(B.subjNum==subs(s) & B.condB==cB);
    xa = todouble(xa); xb = todouble(xb);
    if ~isempty(xa) && ~isempty(xb), v(s) = xa(1)-xb(1); end
end
T = table(subs, v, 'VariableNames', {'subjNum','effect'});
end

function d = todouble(x)
if isstring(x) || iscellstr(x), d = str2double(x); else, d = double(x); end
d = d(~isnan(d));
end

function [rho, p, n, bf10] = run_corr(eegT, band, behT, xlab, ylab, pngPath, isPrimary)
% join EEG band effect with behavioural effect on subjNum, correlate
bf10 = NaN;
if ~ismember(band, string(eegT.Properties.VariableNames))
    fprintf('  [%s] band not found\n', band); rho=NaN;p=NaN;n=0; return;
end
E = eegT(:, {'subjNum', char(band)}); E.Properties.VariableNames{2} = 'eeg';
J = innerjoin(E, behT, 'Keys','subjNum');
ok = ~isnan(J.eeg) & ~isnan(J.effect);
x = J.eeg(ok); y = J.effect(ok); subj = J.subjNum(ok);
n = numel(x);
if n < 4, fprintf('  [%s vs %s] n=%d too few\n', xlab, ylab, n); rho=NaN;p=NaN; return; end

[rho, p]   = corr(x, y, 'type','Spearman');
[rP, pP]   = corr(x, y, 'type','Pearson');
bf10 = jzs_corr_bf(rP, n);      % Bayes factor on the Pearson r
fprintf('  %-32s vs %-30s : n=%2d  rho=%+.3f p=%.4f (Pearson r=%+.3f p=%.4f)  BF10=%.2f\n', ...
        xlab, ylab, n, rho, p, rP, pP, bf10);

% scatter
fig = figure('Color','w','Position',[100 100 560 500],'Visible','off'); hold on;
scatter(x, y, 42, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha',0.7);
% least-squares fit line
b = polyfit(x,y,1); xx = linspace(min(x),max(x),50); plot(xx, polyval(b,xx),'k-','LineWidth',1.5);
xlabel(xlab,'Interpreter','none'); ylabel(ylab,'Interpreter','none');
title(sprintf('n=%d  \\rho=%.2f  p=%.3f  BF_{10}=%.2f', n, rho, p, bf10));
grid on; box on;
try, print(fig, pngPath, '-dpng','-r150'); catch, end
close(fig);

% leave-one-out robustness on the primary test
if isPrimary
    loo = nan(n,1);
    for k = 1:n
        idx = true(n,1); idx(k) = false;
        loo(k) = corr(x(idx), y(idx), 'type','Spearman');
    end
    fprintf('    leave-one-out rho range: [%+.3f, %+.3f]  (full %+.3f)\n', ...
            min(loo), max(loo), rho);
    if sign(min(loo))~=sign(max(loo))
        fprintf('    ** WARNING: sign flips under leave-one-out -- effect is fragile\n');
    end
end
end


function bf10 = jzs_corr_bf(r, n)
% JZS/Jeffreys default Bayes factor for a Pearson correlation, uniform prior
% on rho over (-1,1) (kappa = 1, the JASP default). BF10 = m1/m0 via the
% reduced (Jeffreys) marginal likelihood; the shared constant cancels.
% No toolbox required (2F1 evaluated by its convergent series).
if abs(r) >= 1 || n < 4, bf10 = NaN; return; end
g = @(rho) (1-rho.^2).^((n-1)/2) .* (1-rho.*r).^(-(2*n-3)/2) ...
           .* hyp2f1_series(0.5, 0.5, (2*n-1)/2, (rho.*r + 1)/2);
m1 = 0.5 * integral(g, -1, 1, 'AbsTol',1e-10, 'RelTol',1e-8);
m0 = g(0);
bf10 = m1 / m0;
end

function s = hyp2f1_series(a, b, c, z)
% Gauss 2F1(a,b;c;z) by its convergent series; vectorised over z (|z|<1 here).
s = ones(size(z)); term = ones(size(z));
for k = 0:499
    term = term .* (a+k).*(b+k)./((c+k).*(k+1)) .* z;
    s = s + term;
    if max(abs(term)) < 1e-14, break; end
end
end

function lbl = bf_label(bf10)
% Jeffreys evidence categories, phrased toward whichever hypothesis is favoured.
if bf10 >= 1, x = bf10; dir = 'H1 (correlation)'; else, x = 1/bf10; dir = 'H0 (null)'; end
if     x < 1,   c = 'no evidence';
elseif x < 3,   c = 'anecdotal evidence';
elseif x < 10,  c = 'moderate evidence';
elseif x < 30,  c = 'strong evidence';
elseif x < 100, c = 'very strong evidence';
else,           c = 'extreme evidence';
end
lbl = sprintf('%s for %s', c, dir);
end
