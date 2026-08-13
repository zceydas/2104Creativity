% =====================================================================
% build_QC_table.m
% ---------------------------------------------------------------------
% Rebuild the full QC_summary table by scanning the saved PostICA_*.set
% files on disk. Does NOT reprocess anything -- it reads the metrics that
% are already stored inside each .set (channel mask, sample mask, IC
% classifications, events) and recomputes the residual line-noise numbers
% from the final data. Run this any time after a batch, partial or complete,
% to get an authoritative table that does not depend on which files a given
% run happened to touch.
%
% Recoverable exactly from the .set:
%   n_chan_in, n_chan_removed, chan_removed, pct_samples_removed, data_rank,
%   n_IC_total, n_IC_removed, n_muscle/eye/line/other_removed,
%   n_speak_markers, speech_aware_ICA, line_cleanup, zap_noisefreq,
%   zap_nremove_mean, and the residual line60_after_db / line_flagged_after.
%
% NOT stored in the final .set (set to NaN here):
%   line60_raw_db      -- needs the raw BDF (final data is fully processed)
%   line60_postICA_db  -- pre-cleanup value; for flagged files it is gone
%                         (equals line60_after_db for non-flagged files).
%   If you need those, they are in the per-run logs / the old CSV.
% =====================================================================

clearvars;
eeglab_path = '/usr/local/MATLAB/R2024b/toolbox/eeglab/eeglab2024.2';
addpath(eeglab_path);
eeglab nogui;

% ---- point these at your run ----
OutputBase = '/data2/2104/source_data/outputs/RAT_preprocessing/run_20260810';
OUT_CSV    = fullfile(OutputBase, 'QC_summary.csv');
OUT_MAT    = fullfile(OutputBase, 'QC_summary.mat');

% ---- must match the pipeline ----
ICLabelThresh = [
    NaN NaN;   %Brain
    0.8 1.0;   %Muscle
    0.8 1.0;   %Eye
    NaN NaN;   %Heart
    0.8 1.0;   %Line Noise
    NaN NaN;   %Channel Noise
    NaN NaN];  %Other
SPEAK_CODE   = 203;
PSD_CHANS    = 1:64;
PSD_WINSEC   = 2;
PSD_OVERLAP  = 0.5;
FLAG_LINEFREQ = 60;
FLAG_WINHZ    = 6;
FLAG_DB       = 4;

% ---- find every PostICA set under the output tree ----
sets = dir(fullfile(OutputBase, '**', 'PostICA_*.set'));
if isempty(sets)
    error('No PostICA_*.set files found under %s', OutputBase);
end
fprintf('Found %d PostICA sets. Rebuilding QC table...\n', numel(sets));

QC = struct([]);
for i = 1:numel(sets)
    setFolder = sets(i).folder;
    setName   = sets(i).name;

    % subject / session from the folder path: .../OutputBase/<subject>/<session>
    [sesPath, sessionName] = fileparts(setFolder);
    [~, subjectID]         = fileparts(sesPath);
    stem    = erase(setName, '.set');
    bdfStem = erase(stem, ['PostICA_' subjectID '_' sessionName '_']);

    fprintf('  [%2d/%2d] %s / %s / %s\n', i, numel(sets), subjectID, sessionName, bdfStem);

    try
        EEG = pop_loadset('filename', setName, 'filepath', setFolder);

        % ---- channels ----
        if isfield(EEG.etc, 'clean_channel_mask')
            chMask       = logical(EEG.etc.clean_channel_mask(:))';
            nChanIn      = numel(chMask);
            nChanKept    = sum(chMask);
            nChanRemoved = sum(~chMask);
            if nChanRemoved > 0 && numel(EEG.chanlocs) == nChanIn
                chanRemovedStr = strjoin({EEG.chanlocs(~chMask).labels}, ' ');
            else
                chanRemovedStr = '';
            end
        else
            nChanIn = EEG.nbchan; nChanKept = EEG.nbchan; nChanRemoved = 0; chanRemovedStr = '';
        end
        dataRank = nChanKept;

        % ---- samples removed ----
        if isfield(EEG.etc, 'clean_sample_mask')
            sm = logical(EEG.etc.clean_sample_mask(:))';
            pctSampRemoved = 100 * sum(~sm) / numel(sm);
        else
            pctSampRemoved = 0;
        end

        % ---- IC counts: reconstruct the flagging from stored classifications ----
        nICtotal = NaN; nICremoved = NaN; nMuscle = NaN; nEye = NaN; nLine = NaN; nOther = NaN;
        if isfield(EEG.etc,'ic_classification') && isfield(EEG.etc.ic_classification,'ICLabel')
            icl = EEG.etc.ic_classification.ICLabel;
            if isfield(icl,'orig_classifications') && ~isempty(icl.orig_classifications)
                cls = icl.orig_classifications;
            else
                cls = icl.classifications;   % fallback (kept comps only)
            end
            nICtotal = size(cls,1);
            flagged = false(nICtotal,1);
            for k = 1:size(ICLabelThresh,1)
                if ~any(isnan(ICLabelThresh(k,:)))
                    flagged = flagged | (cls(:,k) >= ICLabelThresh(k,1) & cls(:,k) <= ICLabelThresh(k,2));
                end
            end
            [~, maxClass] = max(cls, [], 2);
            nICremoved = sum(flagged);
            nMuscle = sum(flagged & maxClass==2);
            nEye    = sum(flagged & maxClass==3);
            nLine   = sum(flagged & maxClass==5);
            nOther  = nICremoved - nMuscle - nEye - nLine;
        end

        % ---- speak markers ----
        nSpeak = count_code(EEG.event, SPEAK_CODE);
        speechAware = nSpeak > 0;

        % ---- residual line noise in the FINAL saved data ----
        [fv, pd] = psd_snapshot(EEG, PSD_CHANS, PSD_WINSEC, PSD_OVERLAP);
        [flagAfter, line60_after] = zapline_flag(fv, pd, FLAG_LINEFREQ, FLAG_WINHZ, FLAG_DB);

        % ---- was line-noise cleanup applied? (zapline stores etc.zapline) ----
        if isfield(EEG.etc, 'zapline')
            lineCleanup = "zapline"; lineFlagged = true;
            zapNoiseFreq = NaN; zapNremove = NaN;
            try, zapNoiseFreq = EEG.etc.zapline.config.noisefreqs(1); catch, end
            try, zapNremove   = mean(EEG.etc.zapline.analyticsResults.NremoveFinal(:), 'omitnan'); catch, end
            line60_postICA = NaN;                 % pre-cleanup value not recoverable
        else
            lineCleanup = "none"; lineFlagged = false;
            zapNoiseFreq = NaN; zapNremove = NaN;
            line60_postICA = round(line60_after, 2);  % nothing changed it post-ICA
        end

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
            'n_line_removed',      nLine, ...
            'n_other_removed',     nOther, ...
            'line60_raw_db',       NaN, ...
            'line60_postICA_db',   line60_postICA, ...
            'line_flagged',        lineFlagged, ...
            'line_cleanup',        lineCleanup, ...
            'zap_noisefreq',       round(zapNoiseFreq, 2), ...
            'zap_nremove_mean',    round(zapNremove, 2), ...
            'line60_after_db',     round(line60_after, 2), ...
            'line_flagged_after',  flagAfter, ...
            'n_speak_markers',     nSpeak, ...
            'speech_aware_ICA',    speechAware);
        if isempty(QC), QC = row; else, QC(end+1) = row; end %#ok<SAGROW>

    catch err
        warning('Skipped %s (%s)', setName, err.message);
    end
end

if isempty(QC)
    error('No rows built -- every set failed to load?');
end

T = struct2table(QC);
T = sortrows(T, {'subject','session','file'});
writetable(T, OUT_CSV);
save(OUT_MAT, 'T');
fprintf('\nRebuilt QC table: %d rows -> %s\n', height(T), OUT_CSV);
disp(T);


% =====================================================================
% local functions
% =====================================================================
function n = count_code(events, code)
n = 0;
if isempty(events), return; end
for k = 1:numel(events)
    t = events(k).type;
    if isnumeric(t)
        v = t;
    else
        num = regexp(num2str(t), '\d+', 'match', 'once');
        if isempty(num), continue; end
        v = str2double(num);
    end
    if mod(round(v),256) == code, n = n + 1; end
end
end


function [flag, resid_db] = zapline_flag(f, pdb, noisefreq, winHz, threshDb)
half = winHz/2;
win  = (f >= noisefreq-half) & (f <= noisefreq+half);
if ~any(win), flag = false; resid_db = NaN; return; end
pk = (f >= noisefreq-1) & (f <= noisefreq+1);
w  = pdb(win);
base = mean(w(w <= median(w)));
resid_db = max(pdb(pk)) - base;
flag = resid_db > threshDb;
end


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


function [pxx, f] = local_welch(x, fs, winSec, overlapFrac)
x = double(x(:)');
if numel(x) < 16
    f = [0 fs/2]; pxx = [eps eps]; return;
end
win = max(8, round(winSec * fs));
if win > numel(x)
    win = 2^floor(log2(numel(x)));
    win = max(win, 8);
end
w    = 0.5 - 0.5*cos(2*pi*(0:win-1)/max(1,(win-1)));
U    = sum(w.^2);
step = max(1, round(win * (1 - overlapFrac)));
starts = 1:step:(numel(x) - win + 1);
if isempty(starts), starts = 1; end
nfft = win; nbins = floor(nfft/2) + 1; acc = zeros(1, nbins);
for s = starts
    seg = x(s:s+win-1); seg = seg - mean(seg); seg = seg .* w;
    X = fft(seg, nfft); P = (abs(X(1:nbins)).^2) / (fs * U);
    if nbins > 2, P(2:end-1) = 2*P(2:end-1); end
    acc = acc + P;
end
pxx = acc / numel(starts);
f   = (0:nbins-1) * (fs / nfft);
end
