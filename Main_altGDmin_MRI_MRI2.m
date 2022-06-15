clc;
clear all;close all;
global n1 n2 n mk m  S2 q  kk;
filenames=  {'Pincat.mat','brain_T2_T1.mat','speech_seq.mat','Cardiac_ocmr_data.mat','lowres_speech.mat','FB_ungated.mat'};
%filenames={'Pincat.mat','brain_T2_T1.mat'};


[fid,msg] = fopen('Comparison_error.txt','wt');
[fid2,msg] = fopen('Comparison_sim.txt','wt');
fprintf(fid, '%s(%s) & %s & %s  \n','Dataset','Radial','altGDmin-MRI','altGDmin-MRI2');
fprintf(fid2, '%s(%s) & %s & %s   \n','Dataset','Radial','altGDmin-MRI','altGDmin-MRI2');

for jj = 1:1:numel(filenames)
    S = load(filenames{jj});
    [~,name,~] = fileparts(filenames{jj});
    radial=[4,8,16];
    X_image=double(cell2mat(struct2cell(S)));
    
   
    [n1,n2,q]=size(X_image);
    n=n1*n2;
    X_mat=reshape(X_image,[n,q]);
    
    for ii=1:1:length(radial)
        GD_MLS_time=0;
        GD_MLS_error=0;
        [mask1]=goldencart(n1,n2,q,radial(ii));
        mask = fftshift(fftshift(mask1,1),2);
        Samp_loc=double(find(logical(mask)));
        mask3=reshape(mask,[n, q]);
        mk=[];
        for i=1:1:q
            mk(i)=length(find(logical(mask3(:,i))));
            S2(1:mk(i),i)=double(find(logical(mask3(:,i))));
        end
        m=max(mk);
        Y=zeros(m,q);
        for k=1:1:q
            ksc = reshape( fft2( reshape(X_mat(:,k), [n1 n2]) ), [n,1]) ;
            Y(1:mk(k),k)=double(ksc(S2(1:mk(k),k)));
        end
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AltgdMin + Sparse %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        T=70;
        tic;
        L=[];
        [Xbar_hat,flag,resNE,iter] = cgls(@Afft,@Att, Y,0,1e-36,10);
        Ybar_hat=Afft(Xbar_hat);
        Ybar_hat=reshape(Ybar_hat,[m,q]);
        Yinter=Y-Ybar_hat;
        [Uhat]=initAltGDMin(Yinter);
        [Uhat2, Bhat2]=GDMin_wi(T,Uhat,Yinter);
        xT=Uhat2*Bhat2;
        L(:,1:q)=xT+Xbar_hat;
        
        param.Samp_loc=Samp_loc;
        A = @(z)A_fhp3D(z, Samp_loc,n1,n2,q);
        At = @(z) At_fhp3D(z, Samp_loc, n1,n2,q);
        param.A=A;
        param.At=At;
        param.d = A(X_image(:,:,1:q));
        param.T=TempFFT(3);
        param.lambda_L=0.01;
        
        param.nite=10;
        param.tol=0.0025;
        M=At(param.d);
        M=reshape(M,[n1*n2,q]);
        Lpre=M;
        S=zeros(n1*n2,q);
        param.lambda_S=0.001*max(max(abs(M-L)));
        ite=0;
        while(1)
            ite=ite+1;
            M0=M;
            % sparse update
            S=reshape(param.T'*(SoftThresh(param.T*reshape(M-Lpre,[n1,n2,q]),param.lambda_S)),[n1*n2,q]);
            % data consistency
            resk=param.A(reshape(L+S,[n1,n2,q]))-param.d;
            M=L+S-reshape(param.At(resk),[n1*n2,q]);
            Lpre=L;
            tmp2=param.T*reshape(S,[n1,n2,q]);
            if (ite > param.nite) || (norm(M(:)-M0(:))<param.tol*norm(M0(:))), break;end
        end
        Xhat_MGDS1=L+S;
        Xhat_MGDS=reshape(Xhat_MGDS1,n1,n2,q);
        Time_GD_Sparse=  toc;
        %save('C:\Users\sbabu\Desktop\Result\brain_8\Xhat_MGDS.mat', 'Xhat_MGDS');
        % Time_GD_Sparse= [Time_GD_Sparse, toc];
        Error_GD_Sparse=RMSE_modi(Xhat_MGDS,X_image);
        similarity_index=[];
        for i =1:1:q
            mssim=ssim(abs(Xhat_MGDS(:,:,i)/max(max(Xhat_MGDS(:,:,i)))),abs(X_image(:,:,i)/max(max(X_image(:,:,i)))));
            similarity_index(i)=mssim;
        end
        sim_MGDS=min(similarity_index)
    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AltgdMin + MEC %%%%%%%%%%%%%%%%%%%%
        
        m=max(mk);
        L=[];
        T=70;
        tic;
        [Xbar_hat,flag,resNE,iter] = cgls(@Afft,@Att, Y,0,1e-36,10);
        Ybar_hat=Afft(Xbar_hat);
        Ybar_hat=reshape(Ybar_hat,[m,q]);
        Yinter=Y-Ybar_hat;
        [Uhat]=initAltGDMin(Yinter);
        [Uhat2, Bhat2]=GDMin_wi(T,Uhat,Yinter);
        xT=Uhat2*Bhat2;
        L(:,1:q)=xT+Xbar_hat;
        
        Ymec=Y-Afft(L);
        E_mec=[];
        for kk=1:1:q
            E_mec(:,kk)=cgls_modi(@Afft_modi,@At_modi, Ymec(:,kk) ,0,1e-36,3);
        end
        Xhat_GD_MEC1=L+E_mec;
        Xhat_GD_MEC=reshape(Xhat_GD_MEC1,[n1, n2,q]);
        
        Time_GD_MEC=  toc;
        Error_GD_MEC=RMSE_modi(Xhat_GD_MEC,X_image);
        similarity_index=[];
        for i =1:1:q
            mssim=ssim(abs(Xhat_GD_MEC(:,:,i)/max(max(Xhat_GD_MEC(:,:,i)))),abs(X_image(:,:,i)/max(max(X_image(:,:,i)))));
            similarity_index(i)=mssim;
        end
        sim_GD_MEC=min(similarity_index)
        %save('C:\Users\sbabu\Desktop\Result\brain_8\Xhat_MGD_MEC.mat', 'Xhat_GD_MEC');
        fprintf(fid, '%s(%d) & %8.4f (%5.2f)& %8.4f (%5.2f) \n', name, radial(ii),Error_GD_MEC,Time_GD_MEC,Error_GD_Sparse,Time_GD_Sparse);
        fprintf(fid2, '%s(%d) &  %8.4f (%5.2f)& %8.4f (%5.2f)  \n', name, radial(ii),sim_GD_MEC,Time_GD_MEC,sim_MGDS,Time_GD_Sparse);
 
    end
end
fclose(fid);
fclose(fid2);

function y=SoftThresh(x,p)
y=(abs(x)-p).*x./abs(x).*(abs(x)>p);
y(isnan(y))=0;
end
