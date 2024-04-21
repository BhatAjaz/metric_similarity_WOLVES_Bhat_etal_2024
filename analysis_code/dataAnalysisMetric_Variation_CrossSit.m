%author: Ajaz Bhat ajaz.bhat@ubd.edu.bn
% This script generates an output file
%% Data Analysis File
clear all; close all; 
%Global Variables 
nFeatures=2;scale_factor=8;legendInfo=[];MIN_LOOK_DURATION=200/scale_factor;nFix_limit=10;rmseErr=0; milli2sec=1000;
%Plotting Variables
plotStyle = {'k-o','b-+','g-*','c-x','r-s','m-d','y->','k:o','b:+','g:*','c:x','r:s','m:d','y:>','b:<','w.<'};compStyle=7;
%Output File Save Variables
Measure = {}; Mean = {}; Standard_Error = {}; RMSE_val = [];  MAPE_val = [];
T = table (Measure, Mean, Standard_Error, RMSE_val, MAPE_val);
%%           
%Experiment (Task) Variables
TASK = {'NEAR', 'FAR'};
params = yaml.loadFile("Metric_Variation_8sec.yaml", "ConvertToArray", true); %%params = jsondecode(fileread('Metric_Variation.json'));

nObjects = params.nObjects ; nTrainTrials = params.trainTrial.count; nTestTrials = params.testTrial.count;
TRAIN_DUR = params.trainTrial.duration; TEST_DUR = params.testTrial.duration; 

%% empirical data
emp_Means  = [params.empirical.NEAR.target_mean  params.empirical.NEAR.dist_mean; params.empirical.FAR.target_mean  params.empirical.FAR.dist_mean];
emp_Errors = [params.empirical.NEAR.target_error  params.empirical.NEAR.dist_error; params.empirical.FAR.target_error  params.empirical.FAR.dist_error];
%% test vars
numSubjects=500;
correct_proportion=zeros(2,numSubjects,nTestTrials); % 2 is the number of conditions
targLookTime=zeros(2,numSubjects,nTestTrials);
dstrLookTime=zeros(2,numSubjects,nTestTrials);
LearntWords= NaN(2,numSubjects,nObjects);

%% training vars
corrLookTimeTraining=zeros(2,numSubjects,nTrainTrials);
incorrLookTimeTraining=zeros(2,numSubjects,nTrainTrials);
totnlooks=zeros(2,numSubjects,nTrainTrials);
meanlookdur =zeros(2,numSubjects,nTrainTrials);
TotalLookTime=zeros(2,numSubjects,nTrainTrials);
totlonglookdur = zeros(2,numSubjects,nTrainTrials);
mLookCorrect= zeros(2,numSubjects,nObjects);mLookIncorrect=zeros(2,numSubjects,nObjects);

for iterSim=1:2 
    if iterSim ==1
        label='NEAR';
        simName = 'WPR_8s_NEAR_Metric_Variation_results'%'wPPR_1k15k_1secTest_NEAR_Metric_Variation_2019_results'
        xsit_result = load (simName);
    else
        label='FAR';
        simName = 'WPR_8s_FAR_Metric_Variation_results'%'wPPR_1k15k_1secTest_FAR_Metric_Variation_2019_results'
        xsit_result = load (simName);
    end
    legendInfo{iterSim}= label;
    
    %xsit_result.sim.saveSettings('tesw7.json'); %visdiff ('tesw2.json', 'tesw7.json')
    numSubjects=size(xsit_result.test,1);xx=['Number of Subjects is ',num2str(numSubjects)]; disp(xx);% 

    Look_Smoothening_SY_2AFC;% DATA SMOOTHENING if necessary 

    %% TEST CONDITION ANALYSIS
    
    word_On=floor(params.testTrial.word_On/scale_factor)+1; word_Off= floor(params.testTrial.word_Off/scale_factor);
    vis_On = floor(params.testTrial.visuals_On/scale_factor)+1;vis_Off = floor(params.testTrial.visuals_Off/scale_factor);

    targTimeS=0;distTimeS=0;targTimeW=0;distTimeW=0;
    for subject=1:numSubjects
        lcorrect=0;rcorrect=0;targWord=zeros(nObjects,1);dstrWord=zeros(nObjects,1);
        twoSuccess=zeros(nObjects,1);
        for trt=1:nTestTrials
            lLook= sum( xsit_result.test(subject).historyLt(trt,vis_On:vis_Off));%full trial
            rLook= sum( xsit_result.test(subject).historyRt(trt,vis_On:vis_Off));%
    %         lLook=0; for wint=1:length(word_On); lLook= lLook+sum(xsit_result.test(subject).word_On(wint):word_Off(wint));end %only word-on time-windows
    %         rLook=0; for wint=1:length(word_On); rLook= rLook+sum(xsit_result.test(subject).word_On(wint):word_Off(wint));end %only word-on time-windows

            s1= char(xsit_result.test(subject).test_pair(trt,2*nFeatures+2));
            for kk=1:nObjects    
              if (xsit_result.train(subject).Words{kk} == xsit_result.test(subject).test_pair(trt,2*nFeatures+1))%word index               
                   if ( strcmp(s1,'L')) 
                           targWord(kk)=targWord(kk)+lLook;
                           dstrWord(kk)=dstrWord(kk)+rLook;
                   elseif ( strcmp(s1,'R'))
                           targWord(kk)=targWord(kk)+rLook;
                           dstrWord(kk)=dstrWord(kk)+lLook;
                   else
                           disp('ERROR reading test_pair_char');
                   end
              end
            end
            if ( strcmp(s1,'L')) 
                targLookTime(iterSim,subject,trt)=lLook;
                dstrLookTime(iterSim,subject,trt)=rLook;
            elseif ( strcmp(s1,'R'))
                targLookTime(iterSim,subject,trt)=rLook;
                dstrLookTime(iterSim,subject,trt)=lLook;
            else
                  disp('ERROR reading test_pair char');
            end
        end%% trials loop

        for kk=1:nObjects
            if (targWord(kk)>dstrWord(kk))
                LearntWords(iterSim,subject,kk)=1;
            else
                LearntWords(iterSim,subject,kk)=0;
            end
        end   
    end

    

% disp('t-test statistics between Target and Distractor Looking');
% [h,p,ci,stats] = ttest(mean(targLookTime,2),mean(dstrLookTime,2),'Tail','right')
% disp('t-test statistics for words learnt');
% [h,p,ci,stats] = ttest(sum(LearntWords,2),3,'Tail','right')

% xx=['Avg Looking time per test trial is ',num2str(mean(mean(targLookTime+dstrLookTime))*(scale_factor/milli2sec))]; disp(xx);
% xx=['Looking time to Target per test trial  is ',num2str(mean(mean(targLookTime))*(scale_factor/milli2sec))]; disp(xx);
% xx=['Looking time to Distractor per test trial per is ',num2str(mean(mean(dstrLookTime))*(scale_factor/milli2sec))]; disp(xx);
% xx=['Proportion of time looking correctly (Target/Total) is ',num2str(mean(mean(correct_proportion,2)))]; disp(xx);
% xx=['Number of Strong Learners is ',num2str(sum(goodLearners))]; disp(xx);% Number models classified as strong vs weak
% xx=['Number of Weak Learners is ',num2str(numSubjects-sum(goodLearners))]; disp(xx);% 
% xx=['Looking time to Target per by strong test trial  is ',num2str(mean(mean(targLookTime(goodLearners()==1,:)))*(scale_factor/milli2sec))]; disp(xx);
% xx=['Looking time to Distractor by strong per test trial per is ',num2str(mean(mean(dstrLookTime(goodLearners()==1,:)))*(scale_factor/milli2sec))]; disp(xx);
% xx=['Looking time to Target per by Weak test trial  is ',num2str(mean(mean(targLookTime(goodLearners()==0,:)))*(scale_factor/milli2sec))]; disp(xx);
% xx=['Looking time to Distractor by Weak per test trial per is ',num2str(mean(mean(dstrLookTime(goodLearners()==0,:)))*(scale_factor/milli2sec))]; disp(xx);
% 
% xx=['Avg # of Words Learnt by Strong Learners is ',num2str(mean(sum(LearntWords(goodLearners()==1,:),2)))]; disp(xx);%  
% xx=['Avg # of Words Learnt by Weak Learners is ',num2str(mean(sum(LearntWords(goodLearners()==0,:),2)))]; disp(xx);%  
% xx=['Avg # of Words Learnt is ',num2str(mean(sum(LearntWords(),2)))]; disp(xx);%
% % stanard deviation std(sum(LearntWords(goodLearners()==1,:),2))
% xx=['Avg Looking time per test trial by Strong is ',num2str(mean(mean(targLookTime(goodLearners()==1,:)+dstrLookTime(goodLearners()==1,:)))*(scale_factor/milli2sec))]; disp(xx);
% xx=['Avg Looking time per test trial by Weak is ',num2str(mean(mean(targLookTime(goodLearners()==0,:)+dstrLookTime(goodLearners()==0,:)))*(scale_factor/milli2sec))]; disp(xx);

%% Plot Target vs Distractor looking time during test trial

mT= squeeze(mean(mean(targLookTime(iterSim,:,:)*(scale_factor/milli2sec),3),2));mD= squeeze(mean(mean(dstrLookTime(iterSim,:,:),3),2))*(scale_factor/milli2sec);
eT= std(mean(targLookTime(iterSim,:,:),3))*(scale_factor/milli2sec);eD= std(mean(dstrLookTime(iterSim,:,:),3))*(scale_factor/milli2sec);
sim_means(iterSim,1) = mT; sim_means(iterSim,2) = mD;
errTot(iterSim,1) = eT;      errTot(iterSim,2) = eD;

% figure(300)
% blockNames={'Target'; 'Distractor'};
% %sts = [ mT/(mT+mD) ;  mD/(mT+mD)];
% %errY =[ eT/(mT+mD) ;  eD/(mT+mD)];
% sts = [ mT;  mD];
% errY =[ eT;  eD];
% b=barwitherr(errY, sts);% Plot with errorbars
% set(gca,'xticklabel',blockNames,'fontsize',18);
% %legend('WOLVES Model');
% set(gca,'xticklabel',legendInfo,'fontsize',16);
% ylabel('Proportion Looking');
% ylim([0 0.6]);

%% TRAINING CONDITION ANALYSIS trials analysis
    word_On = floor([params.trainTrial.word1_On params.trainTrial.word2_On]/scale_factor);            
    word_Off = floor([params.trainTrial.word1_Off params.trainTrial.word2_Off ]/scale_factor);word_Len=floor(milli2sec/scale_factor);
    C_word_On = word_On+ floor(500/scale_factor); C_word_Off = word_Off+ floor(500/scale_factor);
    vis_On=1;vis_Off=(TRAIN_DUR/scale_factor); nFix_limit=10;

    for subject=1:numSubjects
        savestate_historyL = xsit_result.train(subject).historyL(:,vis_On:vis_Off);
        savestate_historyR = xsit_result.train(subject).historyR(:,vis_On:vis_Off);    
        % create the off-looking history Vector
        for i=1:nTrainTrials
            for j=1:TRAIN_DUR/scale_factor
               if  (round(savestate_historyL(i,j)) + round(savestate_historyR(i,j))) > 0; savestate_historyO(i,j)=0;               
               else, savestate_historyO(i,j)=1; end
            end
        end

        %%%%% looking during training
        for tr=1:nTrainTrials
            s1=char(xsit_result.train(subject).training_pair(tr,2*nFeatures+3));
            if (strcmp(s1,'P'))% on left
               corrLookTimeTraining(iterSim,subject,tr) = sum(savestate_historyL(tr,C_word_On(1):C_word_Off(1))) ... % first audio presentation, Looking to target object
                + sum(savestate_historyR(tr,C_word_On(2):C_word_Off(2)));% 2nd audio presentation, Looking to target object right side

                incorrLookTimeTraining(iterSim,subject,tr) = sum(savestate_historyR(tr,C_word_On(1):C_word_Off(1))) ... % first audio presentation, Looking wrong way
               + sum(savestate_historyL(tr,C_word_On(2):C_word_Off(2)));% 2nd audio presentation, Looking wrong way
            elseif (strcmp(s1,'X'))
               corrLookTimeTraining(iterSim,subject,tr) = sum(savestate_historyR(tr,C_word_On(1):C_word_Off(1))) ... % first audio presentation, Looking to target object
                + sum(savestate_historyL(tr,C_word_On(2):C_word_Off(2)));% 2nd audio presentation, Looking to target object right side

                incorrLookTimeTraining(iterSim,subject,tr) = sum(savestate_historyL(tr,C_word_On(1):C_word_Off(1))) ... % first audio presentation, Looking wrong way
               + sum(savestate_historyR(tr,C_word_On(2):C_word_Off(2)));% 2nd audio presentation, Looking wrong way            
            end
        end

        %% no# of looks and fixation-durations calculation
        nlooks=zeros(2,nTrainTrials); %L/R
        longlookdur=zeros(2,nTrainTrials);
        all_look_dur=zeros(2,nTrainTrials,nFix_limit);

        for side=1:2     
            if side == 1
                ldata = savestate_historyL;
            else
                ldata = savestate_historyR;
            end
            for tr=1:size(ldata,1)
                prevlook=0;
                templonglookdur=0;
                for time=1:size(ldata,2)
                    if (round(ldata(tr,time)) == 1)
                        if prevlook == 1
                            templonglookdur = templonglookdur+1; 
                        else
                            prevlook = 1;
                            templonglookdur=1;
                        end                    
                    else
                        if prevlook == 1
                            if templonglookdur > (MIN_LOOK_DURATION+5)
                                nlooks(side,tr) = nlooks(side,tr)+1;
                                all_look_dur(side,tr,nlooks(side,tr))= templonglookdur;
                                if templonglookdur > longlookdur(side,tr)
                                    longlookdur(side,tr) = templonglookdur;
                                end
                                templonglookdur=0;
                            end
                            prevlook = 0;
                        end
                    end
                end
                if (round(ldata(tr,time-1)) == 1)
                   if templonglookdur > (MIN_LOOK_DURATION+5)
                        nlooks(side,tr) = nlooks(side,tr)+1;
                        all_look_dur(side,tr,nlooks(side,tr))= templonglookdur;
                        if templonglookdur > longlookdur(side,tr)
                            longlookdur(side,tr) = templonglookdur;
                        end
                   end
                end
            end   
        end

    %     for blockz=1:nObjects
    %         TinA=(nObjects-1)*(blockz-1)+1;
    %         TinB=(nObjects-1)*(blockz);
    %         tLookRepeated(subject,blockz)=sum(sum(savestate_historyL(TinA:TinB,:)));
    %         tLookVarying(subject,blockz)=sum(sum(savestate_historyR(TinA:TinB,:)));
    % 
    %         mLookCorrect(subject,blockz)= mean(corrLookTimeTraining(subject,TinA:TinB));
    %         mLookIncorrect(subject,blockz)= mean(incorrLookTimeTraining(subject,TinA:TinB));
    %     end

        totnlooks(iterSim,subject,:)=sum(nlooks,1);
        meanLukhadur(iterSim,subject,:)=mean(mean(all_look_dur,3),1);
        totlonglookdur(iterSim,subject,:)=max(longlookdur,[],1);    
        TotalLookTime(iterSim,subject,:)=sum(savestate_historyL')+sum(savestate_historyR');    
        meanlookdur(iterSim,subject,:)= TotalLookTime(iterSim,subject,:)./totnlooks(iterSim,subject,:);
         %% calculate entropy in looking on very trial
         total_trialLook_duration=sum(sum(all_look_dur,3),1);
         mean_trialLook_duration=mean(mean(all_look_dur,3),1);
         pdf=zeros(size(all_look_dur));
         variancA=zeros(size(all_look_dur));
         EntropySub(iterSim,subject,:)= 0;
         VarianceSub(iterSim,subject,:)=0;

         for trial=1:size(nlooks,2) 
            for side=1:size(nlooks,1)
                 pdf_side(side,:)=abs(all_look_dur(side,trial,:))./total_trialLook_duration(trial);
                 variance_side(side,:)= (all_look_dur(side,trial,:)./total_trialLook_duration(trial)) .*((all_look_dur(side,trial,:)-mean_trialLook_duration(trial)).^2) ;
            end
            pdf=[pdf_side(1,:) pdf_side(2,:)];
            EntropySub(iterSim,subject,trial)= -1* nansum( pdf(:).*log(pdf(:)) );  
            variancA=[variance_side(1,:) variance_side(2,:)];
            VarianceSub(iterSim,subject,trial)= sum(variancA);
         end     
    end
    asso_threshold = 0.001;
    %% TRACE ANALYSIS
    for subject=1:numSubjects    
        inputMapping1=squeeze(xsit_result.train(subject).hwf(1,:,:));
        inputMapping2=squeeze(xsit_result.train(subject).hwf(2,:,:));
        for kk=1:nObjects
            xx1(kk)=cell2mat(xsit_result.train(subject).Feature1(kk));
            xx2(kk)=cell2mat(xsit_result.train(subject).Feature2(kk));
            yy(kk)=cell2mat(xsit_result.train(subject).Words(kk));
        end
        C_inTr=0;W_inTr=0;
        for kk=1:nObjects
        %%% calculate the number of associations in the trace for each word 
            as_count1(kk)=0; assoc_c=1;
            while assoc_c < size(inputMapping1,1)
                if inputMapping1(assoc_c,yy(kk))>asso_threshold
                    as_count1(kk)=as_count1(kk)+1;
                    while (assoc_c < size(inputMapping1,1)) && (inputMapping1(assoc_c,yy(kk))>=asso_threshold)
                        assoc_c=assoc_c+1;
                    end
                else
                    assoc_c=assoc_c+1;
                end
            end
            as_count2(kk)=0; assoc_c=1;
            while assoc_c < size(inputMapping2,1)
                if inputMapping2(assoc_c,yy(kk))>asso_threshold
                    as_count2(kk)=as_count2(kk)+1;
                    while (assoc_c < size(inputMapping2,1)) && (inputMapping2(assoc_c,yy(kk))>=asso_threshold)
                        assoc_c=assoc_c+1;
                    end
                else
                    assoc_c=assoc_c+1;
                end
            end
            %%% calcuate trace strengths
            a_cv=inputMapping1(xx1(kk),yy(kk));b_cv=inputMapping2(xx2(kk),yy(kk));
            C_inTr= C_inTr+ mean([a_cv b_cv]);
            inputMapping1(xx1(kk),yy(kk))=0;
            inputMapping2(xx2(kk),yy(kk))=0;
            for jj=1:6
                inputMapping1(xx1(kk)-jj,yy(kk))=0; inputMapping1(xx1(kk)+jj,yy(kk))=0;
                inputMapping2(xx2(kk)-jj,yy(kk))=0; inputMapping2(xx2(kk)+jj,yy(kk))=0;
            end
            a_in=inputMapping1(:,yy(kk)); b_in=inputMapping2(:,yy(kk));
            W_inTr = W_inTr + mean([mean(a_in(a_in>asso_threshold)) mean(b_in(b_in>asso_threshold))]);
        end
        Correct_inTrace(iterSim,subject)=C_inTr/nObjects;
        Wrong_inTrace(iterSim, subject)=W_inTr/nObjects;
        InCorr_assocs(iterSim,subject)=mean([as_count1-1 as_count2-1]);
        EntropyTrace(iterSim,subject)= mean( [entropy(inputMapping1) entropy(inputMapping2)] ); 
    end


end


correct_proportion=targLookTime./(targLookTime+dstrLookTime);
disp('t-test statistics between Correct proportions ');
[h,p,ci,stats] = ttest(mean(correct_proportion(1,:,:),3),mean(correct_proportion(2,:,:),3))

 disp('t-test statistics for words learnt');
 [h,p,ci,stats] = ttest(sum(LearntWords(1,:,:),3),sum(LearntWords(2,:,:),3))

%% Data for Output File Saving 
measurement_i = 'Mean correct looking at test';
empirical_mean_i = emp_Means(:,1);
mean_i = sim_means(:,1);
SE_i =   errTot(:,1);
RMSE_i = RMSE(empirical_mean_i, mean_i);MAPE_i = MAPE(empirical_mean_i, mean_i);
%row_i = {measurement_i, num2str(mean_i), num2str(SE_i), RMSE_i, MAPE_i}; T = [T; row_i];
%xx=[measurement_i,' = ', num2str(mean_i)]; disp(xx);
xx=['RMSE = ', num2str(RMSE_i),' and ', 'MAPE = ', num2str(MAPE_i)]; disp(xx);

figure(1005)% Plot Mean correct looking at test
blockNames={'NEAR'; 'FAR'};
b=barwitherr(errTot, sim_means);% Plot with errorbars
set(gca,'xticklabel',blockNames)
ylabel('Mean Looking Time');
legend('Target','Distractor');
grid on
set(gca,'xticklabel',blockNames,'fontsize',18);
set(gca,'xticklabel',legendInfo,'fontsize',16);
%ylim([0 0.6]);
hold on


figure(1006) %% model data CogSci24
sts = [ sim_means(1,1);  sim_means(2,1)];
errY2 =[ errTot(1,1);  errTot(2,1)];

blockNames={'NEAR'; 'FAR'};
b=barwitherr(errY2, sts);% Plot with errorbars
set(gca,'xticklabel',blockNames,'fontsize',18);
legend('WOLVES');
set(gca,'xticklabel',legendInfo,'fontsize',16);
ylabel('mean looking time');
%ylim([0 0.6]);


figure(1007) %% empirical data CogSci24
sts = [ params.empirical.NEAR.target_mean;  params.empirical.FAR.target_mean];
errY2 =[ params.empirical.NEAR.target_error;  params.empirical.FAR.target_error];

blockNames={'NEAR'; 'FAR'};
b=barwitherr(errY2, sts);% Plot with errorbars
set(gca,'xticklabel',blockNames,'fontsize',18);
set(gca,'xticklabel',legendInfo,'fontsize',16);
ylabel('mean looking time');
%ylim([0 0.6]);


figure(1)% Plot total looking time during test trial
%ylim([0 1]);
%x = [1 2];
%bar(x,squeeze(mean(mean(targLookTime,3),2))*scale_factor/milli2sec)
barwitherr([SE(mean(targLookTime(1,:,:),3))*scale_factor/milli2sec; SE(mean(targLookTime(2,:,:),3))*scale_factor/milli2sec], squeeze(mean(mean(targLookTime,3),2))*scale_factor/milli2sec);
hold on
%bar(x,sts)
title('mean looking time at test')
set(gca,'xticklabel',legendInfo,'fontsize',16);
ylabel('Proportion time looking at target');
grid on



figure(2)% Plot total looking time during test trial
ylim([0 1]);
x = [1 2];
barwitherr([std(mean(correct_proportion(1,:,:),3)); std(mean(correct_proportion(2,:,:),3))], squeeze(mean(mean(correct_proportion,3),2)));
%bar(x,squeeze(mean(mean(correct_proportion,3),2)))
hold on
title('At Test')
set(gca,'xticklabel',legendInfo,'fontsize',16);
ylabel('Proportion correct looking');
grid on

figure(3)% Plot total looking time during test trial
%ylim([0 1]);
%x = [1 2];
barwitherr([std(sum(LearntWords(1,:,:),3)); std(sum(LearntWords(2,:,:),3))], squeeze(mean(sum(LearntWords,3),2)));
%bar(x,squeeze(mean(sum(LearntWords,3),2))) 
hold on
%bar(x,sts)
title('At Test')
set(gca,'xticklabel',legendInfo,'fontsize',16);
ylabel('Words Learnt');



figure(5);%Plot looking time during over TEST trials
errorbar(squeeze(mean(targLookTime(1,:,:),2))*scale_factor/milli2sec,(squeeze(std(targLookTime(1,:,:)))*scale_factor/milli2sec)./sqrt(length(targLookTime(1,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(dstrLookTime(1,:,:),2))*scale_factor/milli2sec,(squeeze(std(dstrLookTime(1,:,:)))*scale_factor/milli2sec)./sqrt(length(dstrLookTime(1,:,:))),plotStyle{2+compStyle});%
legend('Target','Distractor');
xlabel('test trial');
ylabel('NEAR total looking time Target vs Distractor');
ylim([0 1]);
title('NEAR')

figure(6);%Plot looking time during over TEST trials
errorbar(squeeze(mean(targLookTime(2,:,:),2))*scale_factor/milli2sec,(squeeze(std(targLookTime(2,:,:)))*scale_factor/milli2sec)./sqrt(length(targLookTime(2,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(dstrLookTime(2,:,:),2))*scale_factor/milli2sec,(squeeze(std(dstrLookTime(2,:,:)))*scale_factor/milli2sec)./sqrt(length(dstrLookTime(2,:,:))),plotStyle{2+compStyle});%
legend('Target','Distractor');
xlabel('test trial');
ylabel('FAR total looking time Target vs Distractor');
ylim([0 1]);
title('FAR')





%% TRAINING CONDITION ANALYSIS trials analysis


figure (10);% Plot entropy in looking fixation durations
errorbar(squeeze(mean(VarianceSub(1,:,:)))*scale_factor/milli2sec,(squeeze(std(VarianceSub(1,:,:))*scale_factor/milli2sec))./sqrt( length( VarianceSub(1,:,:) )),plotStyle{1});% number of fixations/looks over training trials
hold on
errorbar(squeeze(mean(VarianceSub(2,:,:)))*scale_factor/milli2sec,(squeeze(std(VarianceSub(2,:,:))*scale_factor/milli2sec))./sqrt( length( VarianceSub(2,:,:) )),plotStyle{2+compStyle});% number of fixations/looks over training trials
xlabel('per training trial');
ylabel('Variance Near vs Far Condition');
%ylabel('number of fixations/looks Strong learners');
legend('NEAR','FAR');
%ylim([0 2.5]);
%ylim([1.5 3.5])
%hold off
summaF=mean(mean(VarianceSub(1,:,:)));
xx=['Variance NEAR ',num2str(summaF)]; disp(xx);
summaF=mean(mean(VarianceSub(2,:,:)));
xx=['Variance FAR ',num2str(summaF)]; disp(xx);

figure (1001);% Plot entropy in looking fixation durations
errorbar(squeeze(mean(EntropySub(1,:,:))),squeeze(std(EntropySub(1,:,:)))./sqrt( length( EntropySub(1,:,:) )),plotStyle{1});% number of fixations/looks over training trials
hold on
errorbar(squeeze(mean(EntropySub(2,:,:))),squeeze(std(EntropySub(2,:,:)))./sqrt( length( EntropySub(2,:,:) )),plotStyle{2+compStyle});% number of fixations/looks over training trials
xlabel('per training trial');
ylabel('Entropy Near vs far learners');
%ylabel('number of fixations/looks Strong learners');
legend('NEAR','FAR');
summaF=mean(mean(EntropySub(1,:,:)));
xx=['Entropy NEAR ',num2str(summaF)]; disp(xx);
summaF=mean(mean(EntropySub(2,:,:)));
xx=['Entropy FAR ',num2str(summaF)]; disp(xx);

figure(11);%Plot Strong vs Weak looking time during a training trial
errorbar(squeeze(mean(TotalLookTime(1,:,:)))*scale_factor/milli2sec, (squeeze(std(TotalLookTime(1,:,:)))*scale_factor/milli2sec)./sqrt(length(TotalLookTime(1,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(TotalLookTime(2,:,:)))*scale_factor/milli2sec,(squeeze(std(TotalLookTime(2,:,:)))*scale_factor/milli2sec)./sqrt(length(TotalLookTime(2,:,:))),plotStyle{2+compStyle});%
legend('NEAR','FAR');
xlabel('per training trial');
ylabel('total looking time NEAR vs FAR');


figure (12);% Plot number of fixations/looks over training trials
errorbar(squeeze(mean(totnlooks(1,:,:))),(squeeze(std(totnlooks(1,:,:))))./sqrt(length(totnlooks(1,:,:))),plotStyle{1});% number of fixations/looks over training trials
hold on
errorbar(squeeze(mean(totnlooks(2,:,:))),(squeeze(std(totnlooks(2,:,:))))./sqrt(length(totnlooks(2,:,:))),plotStyle{2+compStyle});% number of fixations/looks over training trials
xlabel('per training trial');
ylabel('number of fixations/looks NEAR vs FAR condition');
%ylabel('number of fixations/looks Strong learners');
legend('NEAR','FAR');
summaF=mean(mean(totnlooks(1,:,:)));
xx=['number of fixations/looks NEAR ',num2str(summaF)]; disp(xx);
summaF=mean(mean(totnlooks(2,:,:)));
xx=['number of fixations/looks FAR  ',num2str(summaF)]; disp(xx);


figure (13);%Plot mean look duration of each fixation
errorbar(squeeze(mean(meanlookdur(1,:,:)))*scale_factor/milli2sec, (squeeze(std(meanlookdur(1,:,:)))*scale_factor/milli2sec)./sqrt(length(meanlookdur(1,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(meanlookdur(2,:,:)))*scale_factor/milli2sec,(squeeze(std(meanlookdur(2,:,:)))*scale_factor/milli2sec)./sqrt(length(meanlookdur(2,:,:))),plotStyle{2+compStyle});%
xlabel('per training trial');
ylabel('mean look duration Near Vs Far Condition');
%ylabel('mean look duration Strong learners');
legend('NEAR','FAR');
%hold off
summaF=mean(mean(meanlookdur(1,:,:))*scale_factor)/milli2sec;
xx=['mean look duration NEAR learners ',num2str(summaF)]; disp(xx);
summaF=mean(mean(meanlookdur(2,:,:))*scale_factor)/milli2sec;
xx=['mean look duration FAR weak learners ',num2str(summaF)]; disp(xx);


figure (1301);%Plot mean look duration of each fixation
errorbar(squeeze(mean(meanLukhadur(1,:,:)))*scale_factor/milli2sec, (squeeze(std(meanLukhadur(1,:,:)))*scale_factor/milli2sec)./sqrt(length(meanLukhadur(1,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(meanLukhadur(2,:,:)))*scale_factor/milli2sec,(squeeze(std(meanLukhadur(2,:,:)))*scale_factor/milli2sec)./sqrt(length(meanLukhadur(2,:,:))),plotStyle{2+compStyle});%
xlabel('per training trial (from durations)');
ylabel('mean look duration Near vs Far Condition');
%ylabel('mean look duration Strong learners');
legend('NEAR','FAR');
%hold off
summaF=mean(mean(meanLukhadur(1,:,:))*scale_factor)/milli2sec;
xx=['mean look duration NEAR  (indiv calcs) ',num2str(summaF)]; disp(xx);
summaF=mean(mean(meanLukhadur(2,:,:))*scale_factor)/milli2sec;
xx=['mean look duration FAR (indiv calcs) ',num2str(summaF)]; disp(xx);


figure (14);%Plot duration of longest look per trial
errorbar(squeeze(mean(totlonglookdur(1,:,:)))*scale_factor/milli2sec, (squeeze(std(totlonglookdur(1,:,:)))*scale_factor/milli2sec)./sqrt(length(totlonglookdur(1,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(totlonglookdur(2,:,:)))*scale_factor/milli2sec,(squeeze(std(totlonglookdur(2,:,:)))*scale_factor/milli2sec)./sqrt(length(totlonglookdur(2,:,:))),plotStyle{2+compStyle});%
xlabel('per training trial');
ylabel('duration of longest look Near vs Far learners');

%ylabel('duration of longest look Strong learners');
legend('NEAR','FAR');
%hold off
summaF=mean(mean(totlonglookdur(1,:,:))*scale_factor)/milli2sec;
xx=['Longest look duration NEAR ',num2str(summaF)]; disp(xx);
summaF=mean(mean(totlonglookdur(2,:,:))*scale_factor)/milli2sec;
xx=['Longest look duration FAR ',num2str(summaF)]; disp(xx);


figure(15);%Plot Target vs Distractor looking time during a training trial when words are ON
errorbar(squeeze(mean(corrLookTimeTraining(1,:,:)))*scale_factor/milli2sec, (squeeze(std(corrLookTimeTraining(1,:,:)))*scale_factor/milli2sec)./sqrt(length(corrLookTimeTraining(1,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(incorrLookTimeTraining(1,:,:)))*scale_factor/milli2sec,(squeeze(std(incorrLookTimeTraining(1,:,:)))*scale_factor/milli2sec)./sqrt(length(corrLookTimeTraining(1,:,:))),plotStyle{2+compStyle});%
legend('Correct','Incorrect');
xlabel('Training Trial');
ylabel('NEAR Condition: Looking Time when words are ON');
ylim([0.2 1.75]);

figure(16);%Plot Target vs Distractor looking time during a training trial when words are ON
errorbar(squeeze(mean(corrLookTimeTraining(2,:,:)))*scale_factor/milli2sec, (squeeze(std(corrLookTimeTraining(2,:,:)))*scale_factor/milli2sec)./sqrt(length(corrLookTimeTraining(2,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(incorrLookTimeTraining(2,:,:)))*scale_factor/milli2sec,(squeeze(std(incorrLookTimeTraining(2,:,:)))*scale_factor/milli2sec)./sqrt(length(corrLookTimeTraining(2,:,:))),plotStyle{2+compStyle});%
legend('Correct','Incorrect');
xlabel('Training Trial');
ylabel('FAR Condition: Looking Time when words are ON');
ylim([0.2 1.75]);

figure(1591);%Plot Target vs Distractor looking time during a training trial when words are ON
errorbar(squeeze(mean(mLookCorrect(1,:,:)))*scale_factor/milli2sec, (squeeze(std(mLookCorrect(1,:,:)))*scale_factor/milli2sec)./sqrt(length(mLookCorrect(1,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(mLookIncorrect(1,:,:)))*scale_factor/milli2sec,(squeeze(std(mLookIncorrect(1,:,:)))*scale_factor/milli2sec)./sqrt(length(mLookIncorrect(1,:,:))),plotStyle{2+compStyle});%
legend('Correct','Incorrect');
xlabel('Training Trial');
ylabel('NEAR Condition: Looking Time when words are ON');
ylim([0.2 1.75]);

figure(1592);%Plot Target vs Distractor looking time during a training trial when words are ON
errorbar(squeeze(mean(mLookCorrect(2,:,:)))*scale_factor/milli2sec, (squeeze(std(mLookCorrect(2,:,:)))*scale_factor/milli2sec)./sqrt(length(mLookCorrect(2,:,:))),plotStyle{1});%
hold on
errorbar(squeeze(mean(mLookIncorrect(2,:,:)))*scale_factor/milli2sec,(squeeze(std(mLookIncorrect(2,:,:)))*scale_factor/milli2sec)./sqrt(length(mLookIncorrect(2,:,:))),plotStyle{2+compStyle});%
legend('Correct','Incorrect');
xlabel('Training Trial');
ylabel('FAR Condition: Looking Time when words are ON');
ylim([0.2 1.75]);


summaF=mean(mean(TotalLookTime(1,:,:))) *(scale_factor/milli2sec);
xx=['NEAR Condition: Avg looking time per training trial is ',num2str(summaF)]; disp(xx);

summaF=mean(mean(TotalLookTime(2,:,:))) *(scale_factor/milli2sec);
xx=['FAR Condition: Avg looking time per training trial is ',num2str(summaF)]; disp(xx);




%% TRACE ANALYSIS    
% figure(212121);%Entropy
% blockNames={'NEAR';'FAR'};
% sts = [mean(EntropyTrace(1,:)); mean(EntropyTrace(2,:))  ];
% errY =[std(EntropyTrace(1,:))/sqrt(length(EntropyTrace(1,:))); std(EntropyTrace(2,:))/sqrt(length(EntropyTrace(2,:)))];
% b=barwitherr(errY, sts);% Plot with errorbars
% set(gca,'xticklabel',blockNames,'fontsize',16);
% title ('Entropy in the traces');
% %ylabel('Looking time per test trial');
% %ylim([0 4]);

figure(21);%Entropy
blockNames={'NEAR';'FAR'};
sts = [mean(EntropyTrace(1,:)); mean(EntropyTrace(2,:))  ];
errY =[std(EntropyTrace(1,:)); std(EntropyTrace(2,:))];
b=barwitherr(errY, sts);% Plot with errorbars
set(gca,'xticklabel',blockNames,'fontsize',16);
title ('Entropy in the traces');

figure(22);%My own Entropy: No of incorrect traces
blockNames={'NEAR';'FAR'};
sts = [mean(InCorr_assocs(1,:)); mean(InCorr_assocs(2,:))  ];
errY =[std(InCorr_assocs(1,:))/sqrt(length(InCorr_assocs(1,:))); std(InCorr_assocs(2,:))/sqrt(length(InCorr_assocs(2,:)))];
b=barwitherr(errY, sts);% Plot with errorbars
set(gca,'xticklabel',blockNames,'fontsize',16);
title ('Proportion of incorrect assocs in the traces');
ylabel('# of incorrect assocs per word');

figure(221);%Strength of correct assocs in the traces
blockNames={'NEAR';'FAR'};
sts = [mean(Correct_inTrace(1,:)); mean(Correct_inTrace(2,:))  ];
errY =[std(Correct_inTrace(1,:)); std(Correct_inTrace(2,:))];
b=barwitherr(errY, sts);% Plot with errorbars
set(gca,'xticklabel',blockNames,'fontsize',18);
title ('Strength of correct assocs in the traces');

figure(222);%Strength of correct assocs in the traces
blockNames={'NEAR';'FAR'};
sts = [nanmean(Wrong_inTrace(1,:)); nanmean(Wrong_inTrace(2,:))  ];
errY =[nanstd(Wrong_inTrace(1,:))/sqrt(length(Wrong_inTrace(1,:))); nanstd(Wrong_inTrace(2,:))/sqrt(length(Wrong_inTrace(2,:)))];
b=barwitherr(errY, sts);% Plot with errorbars
set(gca,'xticklabel',blockNames,'fontsize',16);
title ('Strength of INCORRECT assocs in the traces');



%% statistics for Output File Saving 
% mean_i =  novelty_pref_sim_mean;
% SE_i = novelty_pref_sim_SE;
% measurement_i = 'novelty preference';
% RMSE_young = RMSE(empirical_mean_young, mean_i);MAPE_young = MAPE(empirical_mean_young, mean_i);
% xx=['RMSE_young = ', num2str(RMSE_young),' and ', 'MAPE_young = ', num2str(MAPE_young)]; disp(xx);
% RMSE_old = RMSE(empirical_mean_old, mean_i);MAPE_old = MAPE(empirical_mean_old, mean_i);
% xx=['RMSE_old = ', num2str(RMSE_old),' and ', 'MAPE_old = ', num2str(MAPE_old)]; disp(xx);
% row_i = {measurement_i, num2str(mean_i), num2str(SE_i), RMSE_young, MAPE_young,RMSE_old,MAPE_old}; T = [T; row_i];
% 
% 
% %% write table T to output csv file% Name your output file
% writetable(T,[simName  '_Analysis.csv'])


%%%%%%% Association hwf trace analysis
% corrAsocn=zeros(numSubjects,nObjects);
% cS=1;cW=1;
% for subject=1:numSubjects  
%     
%     AsocMat=squeeze(xsit_result.train(subject).hwf(1,:,:));
%     inputMapping=zeros(size(AsocMat));
%      for kk=1:nObjects
%          inputMapping(cell2mat(xsit_result.train(subject).Feature1(kk)),cell2mat(xsit_result.train(subject).Words(kk)))=1;
%      end 
%     
%     for kk=1:nObjects        
%         temp=[];
%         temp=AsocMat(:,cell2mat(xsit_result.train(subject).Words(kk)));  
%         maxAsocnVal(subject,kk) = max(temp);
%         [temp2 in2] = max(temp); 
%         NxtmaxAsocnVal(subject,kk)= max(max (temp(1:max(1,in2-5))), max (temp(min(size(temp),in2+5):size(temp))));
%         %NxtmaxAsocnVal(subject,kk) = max(temp(temp<max(temp)));
%         ratioMax(subject,kk)= maxAsocnVal(subject,kk)./NxtmaxAsocnVal(subject,kk);
%         prodtMR(subject,kk)=ratioMax(subject,kk).*maxAsocnVal(subject,kk);
%         
%         
%         [maxIn(kk), indIn(kk)] = max(inputMapping(:,cell2mat(xsit_result.train(subject).Words(kk))));
%         [maxAs(kk) indAs(kk)] = max(AsocMat(:,cell2mat(xsit_result.train(subject).Words(kk))));       
%         if (abs(indIn(kk)-indAs(kk)) <= 2)%if association is correct i..e same as input?
%            corrAsocn(subject, kk)=1; % wrongAssocn = 6-corrAsocn
%         end
%     end 
% end
% 
% % 
% SLer=[];WLer=[];SNon=[];WNon=[];SLer2=[];WLer2=[];SNon2=[];WNon2=[];
% for subject=1:numSubjects 
%     if(goodLearners(subject)==1)
%         SLer=[SLer maxAsocnVal(subject,LearntWords(subject,:)==1)];
%         SNon=[SNon maxAsocnVal(subject,LearntWords(subject,:)==0)];
%         
%         SLer2=[SLer2 ratioMax(subject,LearntWords(subject,:)==1)];
%         SNon2=[SNon2 ratioMax(subject,LearntWords(subject,:)==0)];
%      elseif (goodLearners(subject)==0)
%          WLer=[WLer maxAsocnVal(subject,LearntWords(subject,:)==1)];
%          WNon=[WNon maxAsocnVal(subject,LearntWords(subject,:)==0)];
%          
%          WLer2=[WLer2 ratioMax(subject,LearntWords(subject,:)==1)];
%          WNon2=[WNon2 ratioMax(subject,LearntWords(subject,:)==0)];
%     end
% end
% if ((size(SLer,2)+size(WLer,2)+size(SNon,2)+size(WNon,2))./numSubjects ~= nObjects), disp('ERROR ERROR ERROR ERROR'), end
% 
% 
%  varb=mean(SLer); xx=['Avg association strength for Learnt words in Strong learners ',num2str(varb)]; disp(xx);
%  varb=mean(WLer); xx=['Avg association strength for Learnt words in Weak learners ',num2str(varb)]; disp(xx);
%  varb=mean(SNon); xx=['Avg association strength for NonLearnt words in Strong learners ',num2str(varb)]; disp(xx);
%  varb=mean(WNon); xx=['Avg association strength for NonLearnt words in Weak learners ',num2str(varb)]; disp(xx);
%   varb=mean(SLer2); xx=['Avg Ratio of 2Maximums for Learnt words in Strong learners ',num2str(varb)]; disp(xx);
%  varb=mean(WLer2); xx=['Avg Ratio of 2Maximums for Learnt words in Weak learners ',num2str(varb)]; disp(xx);
%  varb=mean(SNon2); xx=['Avg Ratio of 2Maximums for NonLearnt words in Strong learners ',num2str(varb)]; disp(xx);
%  varb=mean(WNon2); xx=['Avg Ratio of 2Maximums for NonLearnt words in Weak learners ',num2str(varb)]; disp(xx);
% % %wordsAssoc=sum(corrAsocn,2);
%  varb=mean(sum(corrAsocn,2))/nObjects; xx=['Avg proportion of correctly associated words in Memory ',num2str(varb)]; disp(xx);
% varb=mean(sum(corrAsocn((goodLearners()==1),:),2)); xx=['Avg # of correctly associated words for Strong in Memory ',num2str(varb)]; disp(xx);
% varb=mean(sum(corrAsocn((goodLearners()==0),:),2)); xx=['Avg # of correctly associated words for Weak in Memory ',num2str(varb)]; disp(xx);
% % %Learnt non learnt by strong weak 
% varb=mean(corrWordsStrong); xx=['Avg # of correctly associated words in Memory for Strong counted as Learnt thru looking ',num2str(varb)]; disp(xx);
% std(corrWordsStrong);
% varb=mean(corrWordsWeak); xx=['Avg # of correctly associated words in Memory for Weak counted as Learnt thru looking ',num2str(varb)]; disp(xx);
% std(corrWordsWeak);
% varb=sum((corrAsocn(:,1:6).*LearntWords(:,1:6)));
% xx=['# of subjects with correctly associated word in Memory counted as Learnt thru looking for ',num2str(numSubjects),' subjects is ' num2str(varb)]; disp(xx);


% for subject=1:numSubjects
%     inputMapping1=squeeze(xsit_result.train(subject).hwm_c(1,:,:));
%     inputMapping2=squeeze(xsit_result.train(subject).hwm_c(2,:,:));
%     Repeated_side(subject) = mean([mean(sum(inputMapping1(:,1:50),2))   mean(sum(inputMapping2(:,1:50),2))  ]);
%     Varying_side(subject) = mean([mean(sum(inputMapping1(:,51:100),2))  mean(sum(inputMapping2(:,51:100),2)) ]);
%     
% end
%     
% figure(20132)% Plot Target vs Distractor looking time during test trial
% blockNames={'Repeated'; 'Varying'};
% sts = [  mean(Repeated_side((goodLearners()==1))) mean(Repeated_side((goodLearners()==0)));   mean(Varying_side((goodLearners()==1))) mean(Varying_side((goodLearners()==0)));];
% errY =[  std(Repeated_side((goodLearners()==1))) std(Repeated_side((goodLearners()==0)));   std(Varying_side((goodLearners()==1))) std(Varying_side((goodLearners()==0)));];
% b=barwitherr(errY, sts);% Plot with errorbars
% set(gca,'xticklabel',blockNames);
% legend('Learners', 'Non-Learners');
% title ('Strength of scene memory trace');
% %ylabel('Proportion');
% %ylim([0 0.6]);
% set(gca,'xticklabel',blockNames,'fontsize',16);
% grid on
% for subject=1:numSubjects
% 
% inputMapping1=zeros(306,20);
% inputMapping2=zeros(306,20);
% muddled_pairs=0;
% muddled_mem_val=1;
%     for kk=1:nObjects
%         xx1(kk)=cell2mat(xsit_result.train(subject).Feature1(kk));
%         xx2(kk)=cell2mat(xsit_result.train(subject).Feature2(kk));
%         yy(kk)=cell2mat(xsit_result.train(subject).Words(kk));
%     end
%     for kk=1:nObjects       
%         inputMapping1(xx1(kk),yy(kk))=1;
%         inputMapping2(xx2(kk),yy(kk))=1;
%         for jj=1:8
%             inputMapping1(xx1(kk)+jj-4,yy(kk))=1;
%             inputMapping2(xx2(kk)+jj-4,yy(kk))=1;
%         end   
%     end
%     
%     figure(23)
%     
%     lsp=subplot(1,2,1);
%     %surface(inputMapping1)
%     %hold on
%     [mA, iA] = max(squeeze(xsit_result.train(subject).hwf(1,:,:)));
%     surface(squeeze(xsit_result.train(subject).hwf(1,:,:)));
%     title([num2str(mA)]);
%     shading flat;     
%     
%     rsp= subplot(1,2,2);
%     hold on
%     
%     title('Input');
%     
% %     surface(squeeze(xsit_result.test(subject).hwft(1,:,:)));
% %     title('Test');
% %     shading flat;
%     
% %     if (goodLearners(subject)==1), suptitle(['Strong # ' num2str(subject)]),
% %     elseif (goodLearners(subject)==0), suptitle(['Weak # ' num2str(subject)]), end
%     pause(7);
%     clf;
% end

%% TIME COURSE LOOKING IMAGEMAP LEFT RIGHT  
% subplot(2,1,1);
% lookL = xsit_result.train(1).historyL(28:30,vis_On:vis_Off)';
% lookR = xsit_result.train(1).historyR(28:30,vis_On:vis_Off)';
% vecL = lookL(:);
% vecR = lookR(:);
% d = [vecL, vecR];%size(d)
% area(d);
% %title('Run 1')
% set(gca,'fontsize',12);
% %edit change color
% subplot(2,1,2);
% lookL = xsit_result.train(9).historyL(28:30,vis_On:vis_Off)';
% lookR = xsit_result.train(9).historyR(28:30,vis_On:vis_Off)';
% vecL = lookL(:);
% vecR = lookR(:);
% c = [vecL, vecR];%size(c)
% area(c);
% %title('Run 2')
% set(gca,'fontsize',12);




% % xA=[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
% % %yA=[22/72 9/72 15/72 8/72 28/72 52/72 66/72 71/72 72/72 72/72 72/72];
% % %yA =[3.68 3.3 3.3 3.25 3.9 3.9 4.04 4.74 5.2 5.4 5.5]
% % %yA =[2.02 1.7 1.7 2.1 2.3 2.85 3.16 3 0 0 0]
% % yA= [2.57 1.9 2.08 2.18 2.98 3.6 3.97 4.7 5.2 5.4 5.5]
% % plot (xA,yA)
% % xlabel('memory trace strength');
% % ylabel('Avg words learnt');
% % %ylabel('proportion of Strong Learners');
% % grid on

% % %%% fixations against other measures
% fix_count_All=mean(totnlooks,2);
% TargDis_All=mean((targLookTime./(targLookTime+dstrLookTime)),2);
% Words_All=mean(LearntWords,2);
% T1=fix_count_All(:);
% S1=InCorr_assocs(:); S2=TargDis_All(:); S3=Words_All(:); S4= Correct_inTrace(:)+Wrong_inTrace(:);
% figure (31)
% scatter(T1,S1);
% yb = scatstat1(T1,S1,0,@mean);
% plot(T1,yb,'bo')
% xlabel('fixation count')
% ylabel('# incorrect associations');
% % %[fitobject,gof]=fit(T1,yb,'poly1')
% % %scatter(mean(sync_time,2),mean(targLookTime./(targLookTime+dstrLookTime),2))
% % 
% % figure (33)
% % scatter(T1,S2);
% % %hold on
% % yb = scatstat1(T1,S2,0,@mean);
% % plot(T1,yb,'bo')
% % xlabel('fixation count')
% % ylabel('Prop time looking to target');
% % 
% % figure (33)
% % scatter(T1,S3);
% % %hold on
% % yb = scatstat1(T1,S3,0,@mean);
% % plot(T1,yb,'bo')
% % xlabel('fixation count')
% % ylabel('words learnt');
% % 
% % 
% % figure (34)
% % scatter(T1,S4);
% % yb = scatstat1(T1,S4,0,@mean);
% % plot(T1,yb,'bo')
% % xlabel('fixation count')
% % ylabel('association strength');
% % [fitobject2,gof2]=fit(T1,yb,'poly1');
% % %[h,p,ci,stats] = ttest2(fix_count_All((goodLearners()==1)),fix_count_All((goodLearners()==0)))

