% =====================================================================
% RAT_bandpower_analysis.m
% ---------------------------------------------------------------------
% Spectral band power in the pre-speech "thinking" window of the RAT task,
% compared across drug conditions within participant.
%
% Thinking window per trial:  ideacode (202, prompt onset) -> speakcode
% (203, participant presses to respond). Trials with no 203 (timeout) have
% no window and are skipped.
%
% Conditions come from the allocation file, NOT the session number
% (drug/placebo order varies per participant). Allocation "Session" 1/2/3
% maps to EEG folders ses-0 / ses-1 / ses-2 (a -1 shift).
%
% Input : PostICA_*.set files produced by the preprocessing pipeline.
% Output: per-session x band x channel power table (long CSV), a
%         participant x condition x band summary, and paired placebo-vs-drug
%         stats per band (averaged over channels).
%
% No Signal Processing Toolbox required (Welch implemented locally).
% =====================================================================

clearvars;
eeglab_path = '/usr/local/MATLAB/R2024b/toolbox/eeglab/eeglab2024.2';
addpath(eeglab_path);
eeglab nogui;

% ---- paths ----
OutputBase = '/data2/2104/source_data/outputs/RAT_preprocessing/run_20260810';
ALLOC_XLSX = '/data2/2104/AnalysisScripts/Allocations.xlsx';   % <-- put the file here
RESULTS_DIR = fullfile(OutputBase, 'analysis');
if ~exist(RESULTS_DIR,'dir'), mkdir(RESULTS_DIR); end

% ---- event codes (from the task's RATtrialstructure.m) ----
IDEACODE  = 202;   % prompt onset -> window START
SPEAKCODE = 203;   % press to speak -> window END

% ---- analysis parameters ----
BANDS = { 'delta',[1 4]; 'theta',[4 8]; 'alpha',[8 13]; ...
          'beta',[13 30]; 'gamma',[30 45] };
PSD_CHANS   = 1:64;      % scalp channels
SPEECH_PREP_BUFFER_S = 0.5;  % trim this much BEFORE speakcode: pre-articulatory
                             % motor/jaw prep ramps up before the keypress, so the
                             % analysed window is [ideacode -> speakcode - buffer].
MIN_WIN_S   = 0.5;       % ignore analysed windows shorter than this (too little data)
MAX_WIN_S   = 15;        % guard against a missed 203 producing a huge window
WELCH_WIN_S = 1.0;       % Welch sub-window (short, because thinking windows are short)
WELCH_OVLP  = 0.5;
LOG_POWER   = true;      % analyse 10*log10(power); recommended for band power

% ---- figures ----
FGRID = 1:0.5:45;        % common frequency grid for the spectrum-overlay plot
CONDA = "placebo";       % contrast: condition A ...
CONDB = "drug";          % ... vs condition B (drug = psilocybin 10 mg)
LABELA = "Placebo"; LABELB = "Psilocybin 10 mg";

% ---- sessions to EXCLUDE (from QC review) ----
% {subjectNumber, 'ses-x'} pairs. Edit to match your final keep/drop list.
EXCLUDE = { 16,'ses-1'; 16,'ses-2'; 21,'ses-2'; 19,'ses-2'; 60,'ses-0' };

% ---- allocation: Session(1/2/3) -> EEG folder ses-(0/1/2) ----
alloc = readtable(ALLOC_XLSX);
% expected columns: ParticipantID / Participant ID, Dose, Session
alloc.Properties.VariableNames = matlab.lang.makeValidName(alloc.Properties.VariableNames);
pidCol  = find(contains(lower(alloc.Properties.VariableNames),'partic'),1);
doseCol = find(contains(lower(alloc.Properties.VariableNames),'dose'),1);
sesCol  = find(contains(lower(alloc.Properties.VariableNames),'session'),1);
allocPID  = alloc{:,pidCol};
allocDose = string(alloc{:,doseCol});
allocSes  = alloc{:,sesCol};

% normalise dose labels to placebo / drug / baseline
cond = strings(size(allocDose));
cond(contains(lower(allocDose),'placebo'))  = "placebo";
cond(contains(lower(allocDose),'baseline')) = "baseline";
cond(contains(allocDose,'mg'))              = "drug";     % "10mg/70kg"

% ---- scan PostICA sets ----
sets = dir(fullfile(OutputBase, '**', 'PostICA_*.set'));
fprintf('Found %d PostICA sets.\n', numel(sets));

longRows = {};   % subject, session, cond, band, chan, power, nTrials
sessPSD  = struct('subject',{},'cond',{},'psd',{});   % channel-mean full PSD per session
chanlocsRef = [];   % captured once, for topographies
for i = 1:numel(sets)
    setFolder = sets(i).folder;
    setName   = sets(i).name;
    [sesPath, sessionName] = fileparts(setFolder);
    [~, subjectID]         = fileparts(sesPath);
    subjNum = str2double(regexp(subjectID,'\d+','match','once'));
    subjNum = mod(subjNum,10000);   % 21040001 -> 1  (strip the 2104 study prefix)

    % skip excluded sessions
    if any(cellfun(@(n,s) n==subjNum && strcmp(s,sessionName), EXCLUDE(:,1), EXCLUDE(:,2)))
        fprintf('  [skip QC] %s / %s\n', subjectID, sessionName); continue;
    end

    % map this EEG session to a condition via allocation
    allocSessNum = str2double(regexp(sessionName,'\d+','match','once')) + 1;  % ses-0 -> Session 1
    aRow = find(allocPID==subjNum & allocSes==allocSessNum, 1);
    if isempty(aRow)
        fprintf('  [no alloc] %s / %s -- skipping\n', subjectID, sessionName); continue;
    end
    thisCond = cond(aRow);

    fprintf('  [%2d/%2d] %s / %s -> %s\n', i, numel(sets), subjectID, sessionName, thisCond);

    try
        EEG = pop_loadset('filename', setName, 'filepath', setFolder);
        fs  = EEG.srate;

        % numeric event codes + latencies
        [codes, lats] = event_codes(EEG.event);

        % pair each 202 with the next 203 -> one thinking window per trial
        ideaIx  = find(codes==IDEACODE);
        speakIx = find(codes==SPEAKCODE);
        wins = [];
        for a = 1:numel(ideaIx)
            s0 = lats(ideaIx(a));
            nxtSpeak = speakIx(lats(speakIx) > s0);
            if isempty(nxtSpeak), continue; end     % timeout trial, no speak -> skip
            s1_speak = lats(nxtSpeak(1));                 % actual keypress (window END raw)
            % if an ideacode of the NEXT trial comes before the speak, skip (malformed)
            nxtIdea = ideaIx(lats(ideaIx) > s0);
            if ~isempty(nxtIdea) && lats(nxtIdea(1)) < s1_speak, continue; end
            % trim the speech-prep buffer off the end
            s1 = s1_speak - round(SPEECH_PREP_BUFFER_S*fs);
            dur = (s1 - s0)/fs;                           % length of the ANALYSED window
            if dur < MIN_WIN_S || (s1_speak - s0)/fs > MAX_WIN_S, continue; end
            wins(end+1,:) = [round(s0) round(s1)]; %#ok<AGROW>
        end
        nTrials = size(wins,1);
        if nTrials == 0
            fprintf('      no usable thinking windows -- skipping\n'); continue;
        end

        % band power per channel, averaged across trial windows
        chans = PSD_CHANS(PSD_CHANS <= EEG.nbchan);
        if isempty(chanlocsRef), chanlocsRef = EEG.chanlocs(chans); end
        nB = size(BANDS,1);
        bandPow = nan(numel(chans), nB);      % chan x band, trial-averaged
        gridSum = zeros(1,numel(FGRID)); gridCnt = zeros(1,numel(FGRID));  % full-spectrum accum
        for ci = 1:numel(chans)
            c = chans(ci);
            trialBand = nan(nTrials, nB);
            for w = 1:nTrials
                seg = double(EEG.data(c, wins(w,1):wins(w,2)));
                [pxx, fv] = local_welch(seg, fs, WELCH_WIN_S, WELCH_OVLP);
                for b = 1:nB
                    fb = BANDS{b,2};
                    m  = fv >= fb(1) & fv < fb(2);
                    if ~any(m), continue; end
                    p = mean(pxx(m));                 % mean power density in band
                    trialBand(w,b) = p;
                end
                % accumulate onto the common frequency grid (for the spectrum plot)
                pg = interp1(fv, pxx, FGRID, 'linear', NaN);
                ok = ~isnan(pg);
                gridSum(ok) = gridSum(ok) + pg(ok);
                gridCnt(ok) = gridCnt(ok) + 1;
            end
            % average power across trials, then (optionally) to dB
            mp = mean(trialBand, 1, 'omitnan');
            if LOG_POWER, mp = 10*log10(mp + eps); end
            bandPow(ci,:) = mp;
        end

        % accumulate long rows
        for ci = 1:numel(chans)
            for b = 1:nB
                longRows(end+1,:) = { string(subjectID), string(sessionName), ...
                    thisCond, string(BANDS{b,1}), string(EEG.chanlocs(chans(ci)).labels), ...
                    bandPow(ci,b), nTrials }; %#ok<AGROW>
            end
        end

        % store channel-mean full-spectrum PSD (dB) for the spectrum-overlay plot
        gridMean = gridSum ./ max(gridCnt,1);
        sessPSD(end+1) = struct('subject',string(subjectID), 'cond',thisCond, ...
                                'psd', 10*log10(gridMean + eps)); %#ok<AGROW>

    catch err
        warning('  failed on %s (%s)', setName, err.message);
    end
end

% ---- assemble long table ----
L = cell2table(longRows, 'VariableNames', ...
    {'subject','session','condition','band','channel','power','n_trials'});
writetable(L, fullfile(RESULTS_DIR,'bandpower_long.csv'));
fprintf('\nWrote %d rows -> %s\n', height(L), fullfile(RESULTS_DIR,'bandpower_long.csv'));

% ---- participant x condition x band summary (mean over channels) ----
G = groupsummary(L, {'subject','condition','band'}, 'mean', 'power');
G.Properties.VariableNames{'mean_power'} = 'power';
writetable(G, fullfile(RESULTS_DIR,'bandpower_by_subject_condition.csv'));

% ---- paired placebo vs drug per band (mean over channels within subject) ----
bandsU = unique(L.band,'stable');
statRows = {};
for b = 1:numel(bandsU)
    bn = bandsU(b);
    gp = G(G.band==bn & G.condition=="placebo", {'subject','power'});
    gd = G(G.band==bn & G.condition=="drug",    {'subject','power'});
    J  = innerjoin(gp, gd, 'Keys','subject');   % paired subjects only
    x  = J.power_gp; y = J.power_gd;             % placebo, drug
    n  = numel(x);
    if n < 2, statRows(end+1,:) = {bn,n,NaN,NaN,NaN,NaN}; continue; end %#ok<AGROW>
    d  = y - x;                                  % drug - placebo
    tval = mean(d)/(std(d)/sqrt(n));
    % two-sided p from t distribution (no toolbox: use erf-based approx via tcdf if available)
    if exist('tcdf','file')
        pval = 2*(1 - tcdf(abs(tval), n-1));
    else
        pval = NaN;   % fill in later / use permutation
    end
    dz = mean(d)/std(d);                          % Cohen's dz (paired)
    statRows(end+1,:) = {bn, n, mean(x), mean(y), tval, pval, dz}; %#ok<AGROW>
end
S = cell2table(statRows, 'VariableNames', ...
    {'band','n_pairs','mean_placebo','mean_drug','t','p','dz'});
writetable(S, fullfile(RESULTS_DIR,'placebo_vs_drug_paired.csv'));
fprintf('\n===== Placebo vs Drug (paired, mean over channels) =====\n');
disp(S);
fprintf('Results in: %s\n', RESULTS_DIR);


% =====================================================================
% FIGURES
% =====================================================================
FIGDIR = fullfile(RESULTS_DIR,'figures');
if ~exist(FIGDIR,'dir'), mkdir(FIGDIR); end
condColors = [0.25 0.45 0.85;    % A (placebo) blue
              0.85 0.35 0.25];   % B (drug)    red

% ---- FIG 1: condition-comparison spectrum (mean +/- SEM over subjects) ----
% average each subject's session PSDs within condition, then across subjects
psdA = collect_cond_psd(sessPSD, CONDA);
psdB = collect_cond_psd(sessPSD, CONDB);
if ~isempty(psdA) && ~isempty(psdB)
    fig = figure('Color','w','Position',[100 100 900 560],'Visible','off'); hold on;
    shaded_line(FGRID, psdA, condColors(1,:));
    shaded_line(FGRID, psdB, condColors(2,:));
    for b = 1:size(BANDS,1)
        xline(BANDS{b,2}(1), ':', BANDS{b,1}, 'Color',[.6 .6 .6], ...
              'FontSize',8,'LabelVerticalAlignment','bottom','HandleVisibility','off');
    end
    xlim([FGRID(1) FGRID(end)]); grid on; box on;
    xlabel('Frequency (Hz)'); ylabel('Power (dB)');
    legend({sprintf('%s (n=%d)',LABELA,size(psdA,1)), '', ...
            sprintf('%s (n=%d)',LABELB,size(psdB,1)), ''}, 'Location','northeast');
    title('Thinking-window spectrum: condition comparison (mean \pm SEM)');
    save_fig(fig, fullfile(FIGDIR,'spectrum_condition_comparison'));
end

% ---- FIG 2: per-band paired lines (spaghetti) placebo vs drug ----
bandsU = string(BANDS(:,1))';
fig = figure('Color','w','Position',[100 100 1200 380],'Visible','off');
for b = 1:numel(bandsU)
    subplot(1,numel(bandsU),b); hold on;
    [xa, xb, subj] = paired_band(G, bandsU(b), CONDA, CONDB);
    for k = 1:numel(subj)
        plot([1 2],[xa(k) xb(k)],'-','Color',[.7 .7 .7 .6]);
        plot(1,xa(k),'o','MarkerSize',4,'MarkerFaceColor',condColors(1,:),'MarkerEdgeColor','none');
        plot(2,xb(k),'o','MarkerSize',4,'MarkerFaceColor',condColors(2,:),'MarkerEdgeColor','none');
    end
    if ~isempty(subj)
        plot([1 2],[mean(xa) mean(xb)],'k-','LineWidth',2);
        plot(1,mean(xa),'ks','MarkerFaceColor','k'); plot(2,mean(xb),'ks','MarkerFaceColor','k');
    end
    xlim([.5 2.5]); set(gca,'XTick',[1 2],'XTickLabel',{char(LABELA),char(LABELB)});
    xtickangle(20); title(bandsU(b)); if b==1, ylabel('Power (dB)'); end; grid on; box on;
end
sgtitle('Band power: within-subject placebo vs psilocybin');
save_fig(fig, fullfile(FIGDIR,'bandpower_paired'));

% ---- FIG 3: topography of the drug-minus-placebo difference, per band ----
if ~isempty(chanlocsRef)
    fig = figure('Color','w','Position',[100 100 1200 320],'Visible','off');
    cmax = 0; diffMaps = cell(1,numel(bandsU));
    for b = 1:numel(bandsU)
        diffMaps{b} = topo_diff(L, bandsU(b), CONDA, CONDB, chanlocsRef);
        cmax = max(cmax, max(abs(diffMaps{b}), [], 'omitnan'));
    end
    for b = 1:numel(bandsU)
        subplot(1,numel(bandsU),b);
        try
            topoplot(diffMaps{b}, chanlocsRef, 'maplimits',[-cmax cmax], ...
                     'electrodes','off','style','map');
        catch, end
        title(bandsU(b));
    end
    colormap(redblue(256));
    cb = colorbar('Position',[0.93 0.2 0.012 0.6]); cb.Label.String = 'drug - placebo (dB)';
    sgtitle('Topography: psilocybin \minus placebo (thinking window)');
    save_fig(fig, fullfile(FIGDIR,'topo_drug_minus_placebo'));
end

fprintf('Figures written to: %s\n', FIGDIR);


% =====================================================================
% local functions
% =====================================================================
function P = collect_cond_psd(sessPSD, cnd)
% average each subject's PSDs within a condition, return subjects x freq
if isempty(sessPSD), P = []; return; end
allc = string({sessPSD.cond}); alls = string({sessPSD.subject});
sel  = allc == cnd;
subs = unique(alls(sel), 'stable');
P = [];
for s = 1:numel(subs)
    rows = find(sel & alls==subs(s));
    M = cell2mat({sessPSD(rows).psd}');
    P(end+1,:) = mean(M,1,'omitnan'); %#ok<AGROW>
end
end

function shaded_line(f, M, col)
% mean +/- SEM shaded line for a subjects x freq matrix
m = mean(M,1,'omitnan');
sem = std(M,0,1,'omitnan')./sqrt(sum(~isnan(M),1));
fill([f fliplr(f)], [m+sem fliplr(m-sem)], col, 'FaceAlpha',0.18, ...
     'EdgeColor','none','HandleVisibility','off');
plot(f, m, 'Color', col, 'LineWidth', 2);
end

function [xa, xb, subj] = paired_band(G, bn, cA, cB)
ga = G(G.band==bn & G.condition==cA, {'subject','power'});
gb = G(G.band==bn & G.condition==cB, {'subject','power'});
J  = innerjoin(ga, gb, 'Keys','subject');
xa = J.power_ga; xb = J.power_gb; subj = J.subject;
end

function d = topo_diff(L, bn, cA, cB, chanlocs)
% per-channel mean (cB - cA) paired difference across subjects, in chanlocs order
labels = string({chanlocs.labels});
d = nan(1,numel(labels));
for ci = 1:numel(labels)
    la = L(L.band==bn & L.condition==cA & L.channel==labels(ci), {'subject','power'});
    lb = L(L.band==bn & L.condition==cB & L.channel==labels(ci), {'subject','power'});
    if isempty(la) || isempty(lb), continue; end
    J = innerjoin(la, lb, 'Keys','subject');
    if isempty(J), continue; end
    d(ci) = mean(J.power_lb - J.power_la, 'omitnan');
end
end

function save_fig(fig, stem)
try, print(fig,[stem '.png'],'-dpng','-r150'); catch e, warning(e.message); end
try, savefig(fig,[stem '.fig']); catch e, warning(e.message); end
close(fig);
end

function c = redblue(n)
% simple blue-white-red diverging colormap
if nargin<1, n=256; end
h = floor(n/2);
r = [linspace(0.23,1,h) linspace(1,0.85,n-h)]';
g = [linspace(0.30,1,h) linspace(1,0.30,n-h)]';
b = [linspace(0.75,1,h) linspace(1,0.23,n-h)]';
c = [r g b];
end

function [codes, lats] = event_codes(events)
codes = nan(1,numel(events)); lats = nan(1,numel(events));
for k = 1:numel(events)
    t = events(k).type;
    if isnumeric(t), v = t;
    else, num = regexp(num2str(t),'\d+','match','once'); v = str2double(num);
    end
    if ~isnan(v), codes(k) = mod(round(v),256); end
    lats(k) = events(k).latency;
end
end

function [pxx, f] = local_welch(x, fs, winSec, overlapFrac)
x = double(x(:)');
if numel(x) < 16, f = [0 fs/2]; pxx = [eps eps]; return; end
win = max(8, round(winSec*fs));
if win > numel(x), win = 2^floor(log2(numel(x))); win = max(win,8); end
w = 0.5 - 0.5*cos(2*pi*(0:win-1)/max(1,(win-1))); U = sum(w.^2);
step = max(1, round(win*(1-overlapFrac)));
starts = 1:step:(numel(x)-win+1); if isempty(starts), starts = 1; end
nfft = win; nbins = floor(nfft/2)+1; acc = zeros(1,nbins);
for s = starts
    seg = x(s:s+win-1); seg = seg - mean(seg); seg = seg.*w;
    X = fft(seg,nfft); P = (abs(X(1:nbins)).^2)/(fs*U);
    if nbins>2, P(2:end-1) = 2*P(2:end-1); end
    acc = acc + P;
end
pxx = acc/numel(starts); f = (0:nbins-1)*(fs/nfft);
end
