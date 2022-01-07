%2021年12月6日
%产生间歇采样转发干扰作为分类数据，干噪比30-60dB之间随机；

%% 间歇采样循环转发干扰
close all;clear;clc
j=sqrt(-1);
data_num=1;   %干扰样本数
samp_num=2000;%距离窗点数
fs = 20e6; %采样频率
B = 10e6;  %信号带宽
taup = 20e-6; %信号脉宽
t = linspace(taup/2,taup/2*3,taup*fs);          %时间序列
k = B / taup;
lfm = exp(1j*pi*k*t.^2);          %LFM信号 复包络

SNR=10; %信噪比dB
echo=zeros(data_num,samp_num,3);     %矩阵大小（500,2000,2）
echo_stft=zeros(data_num,100,247,3);  %矩阵大小（500,200,1000,2）
num_label = 2;
label=zeros(1,data_num)+num_label;                         %标签数据,此干扰标签为0

repetion_times_range=[4,3,2];   %重复转发次数
period_range=[5e-6];    %采样脉冲周期   taup / period = 4 或 2，表示采样次数
duty_range=[20,25,33.33];  %占空比 对应转发次数

for m=1:data_num
    %% 目标回波＋噪声
    JNR=3+round(rand(1,1)*20); %干噪比10-20dB
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
    As=10^(SNR/20);%目标回波幅度
    Aj=10^(JNR/20);%干扰回波幅度
    range_tar=1+round(rand(1,1)*1400);
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %噪声+目标回波 目标在距离窗内200点处
    
    %% 采样＋转发
    period_num = 1/8 * length(t); % 取一个周期点数为50
    loop_num = 3;
    loop_list = zeros(1, loop_num*period_num); % 将每次采样后的信号按倒序存入转发列表
    for i=1:loop_num
        current_samp = sp(1+range_tar+(i*(i+1)/2 - 1)*period_num : range_tar+(i*(i+1)/2 - 1)*period_num+period_num);
        loop_list(1+(3-i)*period_num : (3-i+1)*period_num) = current_samp;
        sp(1+range_tar+(i*(i+1)/2 - 1)*period_num + period_num : range_tar+(i*(i+1)/2 - 1)*period_num+period_num+i*period_num) = Aj * loop_list(1+(3-i)*period_num:3*period_num);
    end
    

    sp=sp/max(sp); %归一化
    sp_abs=abs(sp);

    %% 作ISRJ时域波形
%     figure(1)
%     plot(linspace(0,100,2000),sp);
%     set(gca,'FontName','Times New Roman');
%     title("ISRJ")
%     xlabel('Time/μs','FontSize',15);ylabel('Normalized amplitude','FontSize',15)

    %信号实部、虚部分开存入三维张量中
    echo(m,1:2000,1)=real(sp); 
    echo(m,1:2000,2)=imag(sp);
    echo(m,1:2000,3)=sp_abs; 
%     echo(m,1:2000,4)=angle(sp); 

    %% STFT变换
    [S,~,~,~]=spectrogram(sp,32,32-8,200,fs);
    
    S=S/max(max(S));
    S_abs=abs(S);

    %% 作时频图
    figure(3)
    imagesc(linspace(0,100,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
    set(gca,'FontName','Times New Roman');
    title("ISRJ_loop | STFT")
    xlabel('Time/μs','FontSize',15);ylabel('Frequency/MHz','FontSize',15)


    %% 保存实部、虚部、模值
    echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
    echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
    echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs;
%     echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);
 
    
end

% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\echo.mat' ,'echo')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\echo_stft.mat' ,'echo_stft')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\ISRJ_2\label.mat' ,'label')

% t_data=load('D:\CodeSpace\active_jamming_recognition\data\t_data.mat').t_data;
% tf_data=load('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat').tf_data;
% gt_label=load('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat').gt_label;
% % 
% t_data(1+500*(num_label):500*(num_label+1),:,:)=echo; 
% tf_data(1+500*(num_label):500*(num_label+1),:,:,:)=echo_stft; 
% gt_label(1,1+500*(num_label):500*(num_label+1))=label;
% % 
% save('D:\CodeSpace\active_jamming_recognition\data\t_data.mat','t_data')
% save('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat','tf_data')
% save('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat','gt_label')


% figure(1)
% plot(1:400,squa)
% 
% figure(2)
% plot(1:400,squa1)
% figure(3)
% plot(1:2000,sp)
% figure(4)
% [S,~,~,~]=spectrogram(sp,32*2,32*2-1,200,20e6);
% 
% S=S/max(max(S));
% imagesc(1:size(S,1),1:size(S,2),abs(S))




