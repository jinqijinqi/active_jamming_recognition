%2020.11.17
%产生阻塞干扰作为分类数据，50-80Mhz
close all;clear;clc
j=sqrt(-1);
data_num=500;   %干扰样本数
samp_num=2000;%距离窗点数
fs = 20e6; %采样频率
B = 10e6;  %信号带宽
taup = 20e-6; %信号脉宽
t = linspace(taup/2,taup/2*3,taup*fs);          %时间序列
lfm = exp(1j*pi*B/taup*t.^2);          %LFM信号

SNR=0; %信噪比dB
echo=zeros(data_num,samp_num,3);     %矩阵大小（500,2000,2）
echo_stft=zeros(data_num,100,247,3);  %矩阵大小（500,200,1000,2）
num_label = 6;
label=zeros(1,data_num)+num_label;                         %标签数据,此干扰标签为0

for m=1:data_num

    JNR=30+round(rand(1,1)*30); %干噪比30-60dB
    range=round(rand(1,1)*1500);%瞄频噪声干扰起始点
    Bj=50+round(rand(1,1)*30);%瞄频带宽20-40Mhz
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%噪声基底
    sp=sp/std(sp);
    sp1=sp;
    As=10^(SNR/20);%目标回波幅度
    Aj=10^(JNR/20);  %干扰回波幅度

    sp1_fft=fftshift(fft(sp1));
    sp1_fft(1:round(2000*((1-Bj/80)/2)))=0;
    sp1_fft(round(2000*((1-Bj/80)/2))+round(Bj/80*2000):2000)=0;
    sp1=ifftshift(ifft(sp1_fft));
    range_tar=1+round(rand(1,1)*1400);

    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %噪声+目标回波 目标在距离窗内range点处
    sp=sp+Aj*sp1;

    sp=sp/max(max(sp));
    sp_abs=abs(sp);
     figure(3)
    plot(linspace(0,100,2000),sp);xlabel('时间/μs','FontSize',20);ylabel('归一化幅度','FontSize',20)

    echo(m,1:2000,1)=real(sp);
    echo(m,1:2000,2)=imag(sp);
    echo(m,1:2000,3)=sp_abs;
%     echo(m,1:2000,4)=angle(sp); %信号实部、虚部分开存入三维张量中
    [S,~,~,~]=spectrogram(sp,32,32-8,100,20e6);

    S=S/max(max(S));
    S_abs=abs(S);
    figure(4)
    imagesc(linspace(0,100,size(S,1)),linspace(-40,40,size(S,2)),abs(S));
    xlabel('时间/μs','FontSize',20);ylabel('频率/MHz','FontSize',20)


    echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
    echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
    echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs;
%     echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);



end

% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\blocking_jam_6\echo.mat' ,'echo')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\blocking_jam_6\echo_stft.mat' ,'echo_stft')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\blocking_jam_6\label.mat' ,'label')

t_data=load('D:\CodeSpace\active_jamming_recognition\data\t_data.mat').t_data;
tf_data=load('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat').tf_data;
gt_label=load('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat').gt_label;
%
t_data(1+500*(num_label):500*(num_label+1),:,:)=echo;
tf_data(1+500*(num_label):500*(num_label+1),:,:,:)=echo_stft;
gt_label(1,1+500*(num_label):500*(num_label+1))=label;
%
save('D:\CodeSpace\active_jamming_recognition\data\t_data.mat','t_data')
save('D:\CodeSpace\active_jamming_recognition\data\tf_data.mat','tf_data')
save('D:\CodeSpace\active_jamming_recognition\data\gt_label.mat','gt_label')
