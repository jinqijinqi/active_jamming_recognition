%2020.11.17
%����������ƭ����������Ϊ��������,����JNR=30-60dB���ͺ�1~10us
close all;clear;clc
j=sqrt(-1);
data_num=500;   %����������
samp_num=2000;%���봰����
fs = 20e6; %����Ƶ��
B = 10e6;  %�źŴ���
taup = 20e-6; %�ź�����
t = linspace(taup/2,taup/2*3,taup*fs);          %ʱ������
lfm = exp(1j*pi*B/taup*t.^2);          %LFM�ź�

SNR=0; %�����dB
echo=zeros(data_num,samp_num,3);     %�����С��500,2000,2��
echo_stft=zeros(data_num,100,247,3);  %�����С��500,200,1000,2��
num_label = 3;
label=zeros(1,data_num)+num_label;                         %��ǩ����,�˸��ű�ǩΪ0

for m=1:data_num
    JNR=30+round(rand(1,1)*30); %�����30-60dB
    range=20+round(rand(1,1)*180);%��Ŀ���ͺ�1~10us����20-200����
    sp=randn([1,samp_num])+1j*randn([1,samp_num]);%��������
    sp=sp/std(sp);
    As=10^(SNR/20);%Ŀ��ز�����
    Aj=10^(JNR/20);%���Żز�����
    range_tar=1+round(rand(1,1)*1400);
    sp(1+range_tar:length(lfm)+range_tar)=sp(1+range_tar:length(lfm)+range_tar)+As*lfm;  %����+Ŀ��ز� Ŀ���ھ��봰��200�㴦

    sp(1+range_tar+range:length(lfm)+range_tar+range)=sp(1+range_tar+range:length(lfm)+range_tar+range)+Aj*lfm;%����+Ŀ��ز�+���� 
    

    sp=sp/max(max(sp));
    sp_abs=abs(sp);
    figure(3)
    plot(linspace(0,100,2000),sp);
    set(gca,'FontName','Times New Roman');
    xlabel('Time/��s','FontSize',15);ylabel('Normalized amplitude','FontSize',15)
    
    echo(m,1:2000,1)=real(sp); 
    echo(m,1:2000,2)=imag(sp);
    echo(m,1:2000,3)=sp_abs; 
%     echo(m,1:2000,4)=angle(sp);%�ź�ʵ�����鲿�ֿ�������ά������
    [S,~,~,~]=spectrogram(sp,32,32-8,100,20e6);
    
    S=S/max(max(S));
    S_abs=abs(S); 
    figure(4)
    imagesc(linspace(0,100,size(S,1)),linspace(-10,10,size(S,2)),abs(S));
    set(gca,'FontName','Times New Roman');
    xlabel('Time/��s','FontSize',15);ylabel('Frequency/MHz','FontSize',15)

    
    echo_stft(m,1:size(S,1),1:size(S,2),1)=real(S);
    echo_stft(m,1:size(S,1),1:size(S,2),2)=imag(S);
    echo_stft(m,1:size(S,1),1:size(S,2),3)=S_abs;
%     echo_stft(m,1:size(S,1),1:size(S,2),4)=angle(S);
    
end
% 
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\distance_deception_jam_3\echo.mat' ,'echo')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\distance_deception_jam_3\echo_stft.mat' ,'echo_stft')
% save('F:\deep_learning_for_active_jamming_2020.11.16\jamming_data\distance_deception_jam_3\label.mat' ,'label')

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

% figure(3)
% plot(1:2000,sp)
% figure(4)
% [S,~,~,~]=spectrogram(sp,64,64-7,200,20e6);
% 
% S=S/max(max(S));
% imagesc(1:size(S,1),1:size(S,2),abs(S))