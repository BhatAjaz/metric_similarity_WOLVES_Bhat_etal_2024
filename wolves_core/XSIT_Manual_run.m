close all; clear all; 

%% To run a batch of simulations on a HPC/ multicore pc, set variable mode = 2 (below) and use 'parfor' subjects loop in the corresponding experiment file
%To run a gui simulation only, set variable mode = 1 (below) and use 'for' subjects loop in the experiment file
mode = 2; % set to 1 for auto/gui mode, 0 for singlebatch non-gui, and 2 for multicore batch mode (also need to switch to 'parfor' in the experiment file for mode=2)
D1=datetime
%% sample code needed to run simulator on a high performance cluster
HPC = 0;
if HPC
    run('../../../COSIVINA/setpath.m') % add cosivina and jsoblab files to the matlab path
    addpath(genpath('../../WOLVES_PsychReview_2021'));
    cd('../') % move from the wolves_core folder where you'll launch the job to the main folder
    parpool('SlurmProfile1',96) %this will be HPC-specific
end

gui_speed=50; %update gui after every n iterations: ideal values from 1 to 20.
notes = ['8 sec test simulation word presented 4 times'];% notes about any variable changes  
simNamePar = ['WPR_8s_']; % give a name to your simulation.
sim_seed = rng('shuffle'); %get seed for reproducibility, use rng(sim_seed) for reproduction
createComboSim; %%%create the model, for Kachergis et al task (taskvar==7) change to, createComboSimKachergis; 
if sim.loadSettings('wolvesPaperPR.json','changeable') == 0; disp('json file load ERROR!!'); end; % loads the parameters file
createComboGUI;% create and initialize GUI
createComboControls;% create and initialize GUI controls
if (mode == 2), numSubjects = 500; tolerance = 0; % specify the number of simulations/subjects to run. 300 default 
else,   numSubjects = 1; tolerance = 3; simNamePar = ['guiTest']; end % tolerance is used for gui's only to ensure they don't skip equality(==)conditionals

%% Update Memory Build and Decay Parameters
parBuild = 1000; parDecay = 15000;  
setMemoryTraceTimescales(sim, parBuild, parDecay);

%% TURN ON below Noise Parameters for MSHF/Mather novelty tasks (taskvars 14,15,16 below: CD target journal)
%noise_ior_s = 1.3; noise_wm_f = 2;
%setNoveltySpecificNoiseParams(sim, noise_ior_s, noise_wm_f);

%% Choose and Run the Experiment
taskvar = 17; %update taskvar value to simulate correspoding task/experiment from below: default Smith & Yu (2008,11)

if (taskvar==17)
    %% Prediction Task: Metric Variation in Smith & Yu, Dev Sci, 2008 - standard cross-sit
    taskName = 'Metric_Variation';
    TASK_CONDITION_NEAR_ON = 0; %% set to 1 for NEAR condition, 0 for FAR condition 

    %sim naming 
    if TASK_CONDITION_NEAR_ON, simName = [simNamePar,'NEAR_', taskName,'_']; 
    else, simName = [simNamePar,'FAR_', taskName,'_'];end
    Metric_Variation;
    
end

%% Save Simulation Results 
Results_Saving;

disp(D1-datetime);