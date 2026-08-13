clearvars;
cd('/data2/2104')
eeglab_path = '/usr/local/MATLAB/R2024b/toolbox/eeglab/eeglab2024.2'
addpath(genpath('/home/zsayali1/eeglab_plugins/zapline-plus'));
addpath(eeglab_path);
eeglab;

% Line Noise row = [0.8 1.0]: components ICA isolates as line noise are removed.
ICLabel = [
    NaN NaN; %Brain
    0.8 1.0; %Muscle
    0.8 1.0; %Eye
    NaN NaN; %Heart
    0.8 1.0; %Line Noise
    NaN NaN; %Channel Noise
    NaN NaN; %Other
];

% =====================================================================
% TASK EVENT CODES (RAT)  +  SPEECH-AWARE SETTINGS
% =====================================================================
FIXATIONSTART = 200;
FIXATIONEND   = 201;
IDEACODE      = 202;
SPEAK_CODE    = 203;   % speech ONSET
INSIGHT_CODE  = 204;   % insight rating begins -> speech is finished by here
LAGTIME       = 0.012; % marker lag (s); true speech onset assumed slightly earlier

PRE_BUFFER_S  = 0.4;   % additional back-off before 203 for pre-articulatory EMG
FIXED_DUR_S   = 2.0;   % fallback span after 203 if a trial has no matching 204

% --- Channel selection -----------------------------------------------
% Channels 65-70 are BioSemi externals (EXG1-EXG6). Keep scalp only.
N_SCALP      = 64;

% --- EARLY line-noise pass (Step 5, before ICA) ----------------------
% cleanline preserves rank (per-channel sinusoid subtraction), so it is the
% safe pre-ICA choice. Narrow band + NO scanforlines: on clean files this
% barely touches the data and cannot distort it (fixes the earlier regression
% where widening + scanning made good files worse). Zapline is NOT used here
% because its spatial-subspace removal would cost rank and reintroduce the
% ICA muscle-fragmentation we fixed by keeping ICA full-rank.
EARLY_LINEFREQ   = 60;
EARLY_BANDWIDTH  = 2;    % ~59-61 Hz
EARLY_SCAN       = 0;    % assume exactly 60; do not hunt for a peak

% --- Line-noise flag (Zapline-plus coarse detection rule) ------------
% 6 Hz detection window (detectionWinsize) and a 4 dB peak-vs-baseline
% threshold (coarseFreqDetectPowerDiff) -- the published Zapline-plus rule.
FLAG_LINEFREQ = 60;
FLAG_WINHZ    = 6;
FLAG_DB       = 4;

% --- FINAL conditional cleanup (only on flagged files) ---------------
% Zapline-plus adapts per file (auto detects the line, chunk-wise component
% removal) -- the right tool for the strong / drifting line noise cleanline
% cannot handle. Runs AFTER ICA so its rank cost is harmless.
ZAP_MINFREQ = 55;       % constrain auto-detection to the mains neighbourhood
ZAP_MAXFREQ = 65;
% Fallback if the zapline-plus plugin is not installed on this machine:
FALLBACK_LINEFREQS  = [60 120 180];
FALLBACK_BANDWIDTH  = 8;
FALLBACK_SCAN       = 1;

% --- PSD QC settings -------------------------------------------------
PSD_FOI      = [1 120];
PSD_WINSEC   = 2;
PSD_OVERLAP  = 0.5;
PSD_LINEFREQ = 60;
PSD_CHANS    = 1:64;     % scalp channels only
% =====================================================================

% Base directory that CONTAINS the sub_* folders, i.e. ...\source_data
DIR = '/data2/2104/source_data';
OutputBase = '/data2/2104/source_data/outputs/RAT_preprocessing/run_20260810';

ChanLocsDir='/usr/local/MATLAB/R2024b/toolbox/eeglab/eeglab2024.2/plugins/dipfit/standard_BEM/elec/standard_1005.elc';

cd(eeglab_path);
eeglab;

if ~exist(OutputBase, 'dir'), mkdir(OutputBase); end

HAVE_ZAPLINE = exist('clean_data_with_zapline_plus_eeglab_wrapper', 'file') == 2;
if HAVE_ZAPLINE
    disp('zapline-plus found: flagged files will be cleaned with Zapline-plus.');
else
    warning('zapline-plus NOT on path -- flagged files will fall back to wide cleanline.');
end

% batch QC accumulator: one row per processed file
QC = struct([]);

% iterate subjects: folders named sub_XXXXXXXX
subs = dir(fullfile(DIR, 'sub-*'));
subs = subs([subs.isdir]);
subs = subs(~startsWith({subs.name}, '.'));

for xloop = 1:length(subs)

    subjectID = subs(xloop).name;                 % e.g. sub_21040010
    subjectfolder = fullfile(DIR, subjectID);

    % iterate sessions 0 through 2
    for sess = 0:2

        sessionName = sprintf('ses-%d', sess);
        eegFolder = fullfile(subjectfolder, sessionName, 'eeg');
        if ~exist(eegFolder, 'dir')
            eegFolder = fullfile(subjectfolder, sessionName); 
        end

        if ~exist(eegFolder, 'dir')
            continue; 
        end

        % grab the RAT bdf (glob avoids the sub_ vs sub- naming mismatch)
        bdfFiles = dir(fullfile(eegFolder, '*RAT*.bdf'));
        bdfFiles = bdfFiles(~startsWith({bdfFiles.name}, '.'));
        bdfFiles = bdfFiles(~startsWith({bdfFiles.name}, 'EMPTY', 'IgnoreCase', true));
        if isempty(bdfFiles)
            continue;
        end

        Outputfolderpath = fullfile(OutputBase, subjectID, sessionName);
        if ~exist(Outputfolderpath, 'dir')
            mkdir(Outputfolderpath)
        end

        for f = 1:length(bdfFiles)

            bdfName = bdfFiles(f).name;
            bdfStem = erase(bdfName, '.bdf');

            postSetPath = fullfile(Outputfolderpath, ['PostICA_' subjectID '_' sessionName '_' bdfStem '.set']);
            if exist(postSetPath, 'file')
                disp(['Skipping (already processed): ' subjectID ' / ' sessionName ' / ' bdfName]);
                continue;
            end

            disp(['Processing: ' subjectID ' / ' sessionName ' / ' bdfName]);

            % reset per-file PSD snapshots (store spectra, not full data copies)
            psdF = {}; psdP = {}; psdLabel = {};

            %% STEP 1: import data (BDF / BioSemi)
            EEG = pop_biosig(fullfile(eegFolder, bdfName));

            %% Step 3: look up channel locations, keep SCALP channels only
            EEG = pop_chanedit(EEG, 'lookup', ChanLocsDir);
            EEG = pop_select(EEG, 'channel', 1:min(N_SCALP, EEG.nbchan));
            OGEEG = EEG;                    % scalp montage, used to interpolate back later

            % --- PSD snapshot: raw -----------------------------------------
            [pf, pp] = psd_snapshot(EEG, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
            psdF{end+1} = pf; psdP{end+1} = pp; psdLabel{end+1} = '1 raw';
            [~, line60_raw] = zapline_flag(pf, pp, FLAG_LINEFREQ, FLAG_WINHZ, FLAG_DB);

            %% Step 4: Downsample
            EEG = pop_resample(EEG, 512);

            [pf, pp] = psd_snapshot(EEG, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
            psdF{end+1} = pf; psdP{end+1} = pp; psdLabel{end+1} = '2 downsampled';

            %% Step 5: EARLY line noise -- cleanline, narrow + no scan (rank-preserving)
            EEG = pop_cleanline(EEG, 'linefreqs', EARLY_LINEFREQ, ...
                                     'bandwidth', EARLY_BANDWIDTH, ...
                                     'scanforlines', EARLY_SCAN, ...
                                     'sigtype', 'Channels', ...
                                     'newversion', 0);

            [pf, pp] = psd_snapshot(EEG, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
            psdF{end+1} = pf; psdP{end+1} = pp; psdLabel{end+1} = '3 line-noise removed';

            %% Step 6: Remove bad channels (do NOT interpolate yet -- keeps ICA full-rank)
            EEG = pop_clean_rawdata(EEG,...
                'FlatlineCriterion',5,...
                'ChannelCriterion',0.8,...
                'LineNoiseCriterion',4,...
                'Highpass',[0.25 0.75],...
                'BurstCriterion',20,...
                'WindowCriterion',.25, ...
                'WindowCriterionTolerances', [-Inf,7], ...
                'BurstRejection','off',...
                'Distance','Euclidian');

            % ---- capture channel/sample rejection metrics (masks are fresh here)
            if isfield(EEG.etc, 'clean_channel_mask')
                chMask         = logical(EEG.etc.clean_channel_mask(:))';
                nChanIn        = numel(chMask);
                nChanKept      = sum(chMask);
                nChanRemoved   = sum(~chMask);
                chanRemovedStr = strjoin({OGEEG.chanlocs(~chMask).labels}, ' ');
            else
                nChanIn        = OGEEG.nbchan;
                nChanKept      = OGEEG.nbchan;
                nChanRemoved   = 0;
                chanRemovedStr = '';
            end
            if isfield(EEG.etc, 'clean_sample_mask')
                smMask         = logical(EEG.etc.clean_sample_mask(:))';
                pctSampRemoved = 100 * sum(~smMask) / numel(smMask);
            else
                pctSampRemoved = 0;
            end

            dataRank = nChanKept;   % ICA runs on kept, un-referenced channels -> full rank

            [pf, pp] = psd_snapshot(EEG, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
            psdF{end+1} = pf; psdP{end+1} = pp; psdLabel{end+1} = '4 chan reject + ASR';

            PREstudy = pop_saveset(EEG,'filename', ['PreICA_' subjectID '_' sessionName,'_' bdfStem], 'filepath', Outputfolderpath);

            %% Step 7: Run ICA  (SPEECH-AWARE: train on non-speech data), FULL RANK
            EEGtemp = EEG;
            fs = EEGtemp.srate;

            ev = EEGtemp.event;
            codes = nan(1, numel(ev));
            for k = 1:numel(ev)
                t = ev(k).type;
                if isnumeric(t)
                    codes(k) = t;
                else
                    num = regexp(num2str(t), '\d+', 'match', 'once');
                    if ~isempty(num), codes(k) = str2double(num); end
                end
            end
            codes = mod(round(codes), 256);
            lat = [ev.latency];

            speakIdx   = find(codes == SPEAK_CODE);
            insightIdx = find(codes == INSIGHT_CODE);
            nSpeak     = numel(speakIdx);

            if isempty(speakIdx)
                warning('No speak markers (%d) in %s -- running ICA on full data.', ...
                        SPEAK_CODE, bdfName);
                speechAware = false;
                EEGtemp = pop_runica(EEGtemp,'icatype','runica','concatcond','off');
            else
                speechAware = true;
                reg = zeros(numel(speakIdx), 2);
                for s = 1:numel(speakIdx)
                    sLat = lat(speakIdx(s));
                    startLat = sLat - LAGTIME*fs - PRE_BUFFER_S*fs;
                    nextIns  = insightIdx(lat(insightIdx) > sLat);
                    if isempty(nextIns)
                        endLat = sLat + FIXED_DUR_S*fs;
                    else
                        endLat = lat(nextIns(1));
                    end
                    reg(s,:) = [startLat, endLat];
                end
                reg(:,1) = max(round(reg(:,1)), 1);
                reg(:,2) = min(round(reg(:,2)), EEGtemp.pnts);
                reg = reg(reg(:,2) > reg(:,1), :);

                reg = sortrows(reg);
                speechRegions = reg(1,:);
                for r = 2:size(reg,1)
                    if reg(r,1) <= speechRegions(end,2)+1
                        speechRegions(end,2) = max(speechRegions(end,2), reg(r,2));
                    else
                        speechRegions(end+1,:) = reg(r,:); 
                    end
                end

                EEGtrain = eeg_eegrej(EEGtemp, speechRegions);
                EEGtrain = pop_runica(EEGtrain,'icatype','runica','concatcond','off');

                EEGtemp.icaweights  = EEGtrain.icaweights;
                EEGtemp.icasphere   = EEGtrain.icasphere;
                EEGtemp.icawinv     = EEGtrain.icawinv;
                EEGtemp.icachansind = EEGtrain.icachansind;
                EEGtemp = eeg_checkset(EEGtemp, 'ica');
                clear EEGtrain
            end

            EEGtemp = pop_iclabel(EEGtemp,'default');
            EEGtemp = pop_icflag(EEGtemp, ICLabel);

            rej         = logical(EEGtemp.reject.gcompreject(:));
            nICtotal    = numel(rej);
            nICremoved  = sum(rej);
            [~, maxClass] = max(EEGtemp.etc.ic_classification.ICLabel.classifications, [], 2);
            nMuscle     = sum(rej & maxClass == 2);
            nEye        = sum(rej & maxClass == 3);
            nLineIC     = sum(rej & maxClass == 5);
            nOtherRej   = nICremoved - nMuscle - nEye - nLineIC;

            fprintf('  %s / %s: rank %d | removed %d/%d chan, %d/%d ICs (%d musc, %d eye, %d line)\n', ...
                    subjectID, sessionName, dataRank, nChanRemoved, nChanIn, ...
                    nICremoved, nICtotal, nMuscle, nEye, nLineIC);

            classifications = EEGtemp.etc.ic_classification.ICLabel.classifications;
            EEGtemp = pop_subcomp(EEGtemp,[],0);
            EEGtemp.etc.ic_classification.ICLabel.orig_classifications = classifications;

            [pf, pp] = psd_snapshot(EEGtemp, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
            psdF{end+1} = pf; psdP{end+1} = pp; psdLabel{end+1} = '5 post-ICA';

            %% Step 8: Interpolate removed channels back (AFTER ICA)
            EEGtemp = pop_interp(EEGtemp, eeg_mergelocs(OGEEG.chanlocs), 'spherical');

            %% Step 9: Rereference (average), AFTER interpolation
            EEGtemp = pop_reref(EEGtemp, []);

            [pf, pp] = psd_snapshot(EEGtemp, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
            psdF{end+1} = pf; psdP{end+1} = pp; psdLabel{end+1} = '6 interp + reref';
            [flagFinal, line60_final] = zapline_flag(pf, pp, FLAG_LINEFREQ, FLAG_WINHZ, FLAG_DB);

            %% Step 10: CONDITIONAL line-noise cleanup -- only if still flagged
            zapApplied      = false;
            zapMethod       = "none";
            zapNoiseFreq    = NaN;
            zapNremoveMean  = NaN;
            line60_after    = line60_final;
            flagAfter       = flagFinal;

            if flagFinal
                if HAVE_ZAPLINE
                    zapMethod = "zapline";
                    zapCfg = struct('noisefreqs', [], ...      % auto-detect the line...
                                    'minfreq',    ZAP_MINFREQ, ... % ...within the mains band
                                    'maxfreq',    ZAP_MAXFREQ, ...
                                    'plotResults', 0);
                    EEGtemp.icaact = [];
                    try
                        EEGtemp = clean_data_with_zapline_plus_eeglab_wrapper(EEGtemp, zapCfg);
                        zapApplied = true;
                        try, zapNoiseFreq   = EEGtemp.etc.zapline.config.noisefreqs(1); catch, end
                        try, zapNremoveMean = mean(EEGtemp.etc.zapline.analyticsResults.NremoveFinal(:), 'omitnan'); catch, end
                    catch err
                        warning('Zapline failed on %s (%s) -- falling back to cleanline.', bdfName, err.message);
                        zapMethod = "cleanline_fallback";
                        EEGtemp.icaact = [];
                        EEGtemp = pop_cleanline(EEGtemp, 'linefreqs', FALLBACK_LINEFREQS, ...
                                                'bandwidth', FALLBACK_BANDWIDTH, 'scanforlines', FALLBACK_SCAN, ...
                                                'sigtype', 'Channels', 'newversion', 0);
                        zapApplied = true;
                    end
                else
                    zapMethod = "cleanline_fallback";
                    EEGtemp.icaact = [];
                    EEGtemp = pop_cleanline(EEGtemp, 'linefreqs', FALLBACK_LINEFREQS, ...
                                            'bandwidth', FALLBACK_BANDWIDTH, 'scanforlines', FALLBACK_SCAN, ...
                                            'sigtype', 'Channels', 'newversion', 0);
                    zapApplied = true;
                end

                [pf, pp] = psd_snapshot(EEGtemp, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
                psdF{end+1} = pf; psdP{end+1} = pp; psdLabel{end+1} = ['7 ' char(zapMethod)];
                [flagAfter, line60_after] = zapline_flag(pf, pp, FLAG_LINEFREQ, FLAG_WINHZ, FLAG_DB);

                if flagAfter, statusStr = 'still flagged'; else, statusStr = 'cleared'; end
                fprintf('    line flag: 60 Hz %.1f dB -> %s -> %.1f dB (%s)\n', ...
                        line60_final, char(zapMethod), line60_after, statusStr);
            end

            POSTstudy = pop_saveset(EEGtemp,'filename', ['PostICA_' subjectID '_' sessionName, '_' bdfStem], 'filepath',Outputfolderpath);

            % --- overlay all stages and save PNG + FIG ---------------------
            pngPath = fullfile(Outputfolderpath, ['PSD_' subjectID '_' sessionName '_' bdfStem '.png']);
            figPath = fullfile(Outputfolderpath, ['PSD_' subjectID '_' sessionName '_' bdfStem '.fig']);
            plot_and_save_psd(psdF, psdP, psdLabel, PSD_FOI, PSD_LINEFREQ, pngPath, figPath);

            % --- if line noise flagged, show WHICH channels carry it --------
            if flagFinal
                chanPng = fullfile(Outputfolderpath, ['PSDchan_' subjectID '_' sessionName '_' bdfStem '.png']);
                plot_channelwise_psd(EEGtemp, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP, ...
                                     PSD_FOI, PSD_LINEFREQ, chanPng);
            end

            % --- append this file's QC row ---------------------------------
            row = struct( ...
                'subject',             string(subjectID), ...
                'session',             string(sessionName), ...
                'file',                string(bdfStem), ...
                'n_chan_in',           nChanIn, ...
                'n_chan_removed',      nChanRemoved, ...
                'chan_removed',        string(chanRemovedStr), ...
                'pct_samples_removed', round(pctSampRemoved, 2), ...
                'data_rank',           dataRank, ...
                'n_IC_total',          nICtotal, ...
                'n_IC_removed',        nICremoved, ...
                'n_muscle_removed',    nMuscle, ...
                'n_eye_removed',       nEye, ...
                'n_line_removed',      nLineIC, ...
                'n_other_removed',     nOtherRej, ...
                'line60_raw_db',       round(line60_raw, 2), ...
                'line60_postICA_db',   round(line60_final, 2), ...
                'line_flagged',        flagFinal, ...
                'line_cleanup',        zapMethod, ...
                'zap_noisefreq',       round(zapNoiseFreq, 2), ...
                'zap_nremove_mean',    round(zapNremoveMean, 2), ...
                'line60_after_db',     round(line60_after, 2), ...
                'line_flagged_after',  flagAfter, ...
                'n_speak_markers',     nSpeak, ...
                'speech_aware_ICA',    speechAware);
            if isempty(QC), QC = row; else, QC(end+1) = row; end %#ok<SAGROW>

            writetable(struct2table(QC), fullfile(OutputBase, 'QC_summary.csv'));

        end
        
    end
end

% =====================================================================
% Final QC table
% =====================================================================
if ~isempty(QC)
    T = struct2table(QC);
    writetable(T, fullfile(OutputBase, 'QC_summary.csv'));
    save(fullfile(OutputBase, 'QC_summary.mat'), 'T');
    fprintf('\n===== QC SUMMARY (%d files) =====\n', height(T));
    disp(T);
    nFlag  = sum([QC.line_flagged]);
    nAfter = sum([QC.line_flagged_after]);
    fprintf('%d file(s) flagged for line noise post-ICA; %d still flagged after cleanup.\n', nFlag, nAfter);
    fprintf('Saved: %s\n', fullfile(OutputBase, 'QC_summary.csv'));
else
    disp('No files were processed this run (all skipped or none found).');
end


% =====================================================================
% LOCAL FUNCTIONS
% =====================================================================
function [f, pdb] = psd_snapshot(EEG, chans, winSec, overlap)
data = EEG.data;
if ndims(data) == 3, data = reshape(data, size(data,1), []); end
if isempty(chans), chans = 1:size(data,1); end
chans = chans(chans >= 1 & chans <= size(data,1));
psdSum = []; f = [];
for c = chans
    [pxx, fv] = local_welch(double(data(c,:)), EEG.srate, winSec, overlap);
    if isempty(psdSum), psdSum = zeros(1, numel(pxx)); f = fv; end
    psdSum = psdSum + pxx;
end
pdb = 10*log10(psdSum / numel(chans) + eps);
end


function [flag, resid_db] = zapline_flag(f, pdb, noisefreq, winHz, threshDb)
% Zapline-plus coarse line-noise rule on a channel-averaged log-PSD:
% peak power at noisefreq vs the 'center' (baseline) power in a winHz window,
% flagged if the difference exceeds threshDb (coarseFreqDetectPowerDiff).
half = winHz/2;
win  = (f >= noisefreq-half) & (f <= noisefreq+half);
if ~any(win), flag = false; resid_db = NaN; return; end
pk = (f >= noisefreq-1) & (f <= noisefreq+1);
w  = pdb(win);
base = mean(w(w <= median(w)));   % baseline from lower half of the window (excludes peak)
resid_db = max(pdb(pk)) - base;
flag = resid_db > threshDb;
end


function plot_channelwise_psd(EEG, chans, winSec, overlap, foi, linefreq, pngPath)
% Every channel faint + mean bold; the 5 channels with most power at linefreq
% coloured and named. Answers "is this 2 bad electrodes or the whole cap?".
data = EEG.data;
if ndims(data) == 3, data = reshape(data, size(data,1), []); end
if isempty(chans), chans = 1:size(data,1); end
chans = chans(chans >= 1 & chans <= size(data,1));
psdAll = []; f = [];
for i = 1:numel(chans)
    [pxx, fv] = local_welch(double(data(chans(i),:)), EEG.srate, winSec, overlap);
    if isempty(psdAll), psdAll = zeros(numel(chans), numel(pxx)); f = fv; end
    psdAll(i,:) = 10*log10(pxx + eps);
end
m = (f >= foi(1)) & (f <= foi(2));
fig = figure('Color','w','Position',[100 100 950 580],'Visible','off'); hold on;
plot(f(m), psdAll(:,m).', 'Color', [0.7 0.7 0.7 0.5]);
hMean = plot(f(m), mean(psdAll(:,m),1), 'k', 'LineWidth', 2);
pk = (f >= linefreq-1.5) & (f <= linefreq+1.5);
[~, ord] = sort(max(psdAll(:,pk), [], 2), 'descend');
nTop = min(5, numel(ord)); cols = lines(nTop);
hTop = gobjects(1, nTop); lbl = cell(1, nTop);
for i = 1:nTop
    ci = ord(i);
    hTop(i) = plot(f(m), psdAll(ci,m), 'Color', cols(i,:), 'LineWidth', 1.2);
    if chans(ci) <= numel(EEG.chanlocs) && ~isempty(EEG.chanlocs(chans(ci)).labels)
        lbl{i} = EEG.chanlocs(chans(ci)).labels;
    else
        lbl{i} = sprintf('ch%d', chans(ci));
    end
end
xline(linefreq, ':', sprintf('%d Hz', linefreq), 'Color',[0.55 0.55 0.55], ...
      'FontSize', 8, 'HandleVisibility','off');
legend([hMean hTop], [{'channel mean'} lbl], 'Location','northeast','Interpreter','none');
xlim(foi); grid on; box on;
xlabel('Frequency (Hz)'); ylabel('Power (dB, 10\cdotlog_{10} \muV^2/Hz)');
[~, stem] = fileparts(pngPath);
title([stem '  (worst channels at line freq)'], 'Interpreter','none');
try, print(fig, pngPath, '-dpng', '-r150'); catch e, warning('PNG save failed: %s', e.message); end
close(fig);
end


function plot_and_save_psd(psdF, psdP, labels, foi, linefreq, pngPath, figPath)
fig = figure('Color', 'w', 'Position', [100 100 950 580], 'Visible', 'off'); hold on;
cols = lines(numel(psdF)); h = gobjects(1, numel(psdF));
for i = 1:numel(psdF)
    fi = psdF{i}; pi_ = psdP{i};
    m  = (fi >= foi(1)) & (fi <= foi(2));
    h(i) = plot(fi(m), pi_(m), 'Color', cols(i,:), 'LineWidth', 1.5);
end
if ~isempty(linefreq) && linefreq > 0
    for hz = linefreq:linefreq:foi(2)
        xline(hz, ':', sprintf('%d Hz', hz), 'Color', [0.55 0.55 0.55], ...
              'LabelVerticalAlignment', 'bottom', 'FontSize', 8, 'HandleVisibility', 'off');
    end
end
set(gca, 'XScale', 'linear'); xlim(foi); grid on; box on;
xlabel('Frequency (Hz)'); ylabel('Power (dB, 10\cdotlog_{10} \muV^2/Hz)');
legend(h, labels, 'Location', 'northeast', 'Interpreter', 'none');
[~, stem] = fileparts(pngPath);
title(stem, 'Interpreter', 'none');
try, print(fig, pngPath, '-dpng', '-r150'); catch e, warning('PNG save failed: %s', e.message); end
try, savefig(fig, figPath);              catch e, warning('FIG save failed: %s', e.message); end
close(fig);
end


function [pxx, f] = local_welch(x, fs, winSec, overlapFrac)
% Minimal one-sided Welch PSD (Hann window), no toolbox required.
% Guards against segments too short to form a window (e.g. heavy ASR rejection).
x = double(x(:)');
if numel(x) < 16                       % too short for a meaningful PSD
    f = [0 fs/2]; pxx = [eps eps]; return;
end
win = max(8, round(winSec * fs));
if win > numel(x)
    win = 2^floor(log2(numel(x)));     % largest power-of-2 window that fits
    win = max(win, 8);
end
w    = 0.5 - 0.5*cos(2*pi*(0:win-1)/max(1,(win-1)));
U    = sum(w.^2);
step = max(1, round(win * (1 - overlapFrac)));
starts = 1:step:(numel(x) - win + 1);
if isempty(starts), starts = 1; end
nfft  = win; nbins = floor(nfft/2) + 1; acc = zeros(1, nbins);
for s = starts
    seg = x(s:s+win-1); seg = seg - mean(seg); seg = seg .* w;
    X = fft(seg, nfft); P = (abs(X(1:nbins)).^2) / (fs * U);
    if nbins > 2, P(2:end-1) = 2*P(2:end-1); end
    acc = acc + P;
end
pxx = acc / numel(starts);
f   = (0:nbins-1) * (fs / nfft);
end