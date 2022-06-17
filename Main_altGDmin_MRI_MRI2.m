clc;
clear all;close all;
global n1 n2 n mk m  S2 q  kk;
%filenames=  {'Pincat.mat','brain_T2_T1.mat','speech_seq.mat','Cardiac_ocmr_data.mat','lowres_speech.mat','FB_ungated.mat'};
%filenames={'Pincat.mat','brain_T2_T1.mat'};
filenames={'brain_T2_T1.mat'};

[fid,msg] = fopen('Comparison_error.txt','wt');
[fid2,msg] = fopen('Comparison_sim.txt','wt');
fprintf(fid, '%s(%s) & %s & %s  \n','Dataset','Radial','altGDmin-MRI','altGDmin-MRI2');
fprintf(fid2, '%s(%s) & %s & %s   \n','Dataset','Radial','altGDmin-MRI','altGDmin-MRI2');

for jj = 1:1:numel(filenames)
    Ehat = load(filenames{jj});
    [~,name,~] = fileparts(filenames{jj});
    radial=[4,8,16];
    X_image=double(cell2mat(struct2cell(Ehat)));
    
   
    [n1,n2,q]=size(X_image);
    n=n1*n2;
    X_star=reshape(X_image,[n,q]);
    
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
            ksc = reshape( fft2( reshape(X_star(:,k), [n1 n2]) ), [n,1]) ;
            Y(1:mk(k),k)=double(ksc(S2(1:mk(k),k)));
        end
             
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AltgdMin + Sparse %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        T=70;
        tic;
        L=[];
        [zbar_hat,flag,resNE,iter] = cgls(@Afft,@Att, Y,0,1e-36,10);
        Ytemp=reshape(Afft(zbar_hat),[m,q]);
        Ybar=Y-Ytemp;
        [Uhat]=initAltGDMin(Ybar);
        [Uhat2, Bhat2]=AltGDmin(T,Uhat,Ybar);
        X_hat=Uhat2*Bhat2;
       
        
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
        Ehat=zeros(n1*n2,q);
        L(:,1:q)=X_hat+zbar_hat;
        param.lambda_S=0.001*max(max(abs(M-L)));
        ite=0;
       
        while(1)
            ite=ite+1;
            M0=M;
            Ehat=reshape(param.T'*(SoftThresh(param.T*reshape(M-Lpre,[n1,n2,q]),param.lambda_S)),[n1*n2,q]);
            resk=param.A(reshape(L+Ehat,[n1,n2,q]))-param.d;
            M=L+Ehat-reshape(param.At(resk),[n1*n2,q]);
            Lpre=L;
            tmp2=param.T*reshape(Ehat,[n1,n2,q]);
            if (ite > param.nite) || (norm(M(:)-M0(:))<param.tol*norm(M0(:))), break;end
        end
        Zhat=L+Ehat;
        Zhat_MRI2=reshape(Zhat,n1,n2,q);
        Time_MRI2=  toc;
        %save('C:\Users\sbabu\Desktop\Result\brain_8\Xhat_MGDS.mat', 'Xhat_MGDS');
        Error_MRI2=RMSE_modi(Zhat_MRI2,X_image);
        similarity_index=[];
        for i =1:1:q
            mssim=ssim(abs(Zhat_MRI2(:,:,i)/max(max(Zhat_MRI2(:,:,i)))),abs(X_image(:,:,i)/max(max(X_image(:,:,i)))));
            similarity_index(i)=mssim;
        end
        sim_MRI2=min(similarity_index)
    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AltgdMin + MEC %%%%%%%%%%%%%%%%%%%%
        
        m=max(mk);
        L=[];
        T=70;
        tic;
        
        
        [zbar_hat,flag,resNE,iter] = cgls(@Afft,@Att, Y,0,1e-36,10);
        Ytemp=reshape(Afft(zbar_hat),[m,q]);
        Ybar=Y-Ytemp;
        [Uhat]=initAltGDMin(Ybar);
        [Uhat2, Bhat2]=AltGDmin(T,Uhat,Ybar);
        X_hat=Uhat2*Bhat2;
        
        Yhat_hat=Y-Afft(X_hat+zbar_hat);
        Ehat=[];
        for kk=1:1:q
            Ehat(:,kk)=cgls_modi(@Afft_modi,@At_modi, Yhat_hat(:,kk) ,0,1e-36,3);
        end
        Zhat=X_hat+zbar_hat+Ehat;
        Zhat_MRI=reshape(Zhat,[n1, n2,q]);
        
        Time_MRI=  toc;
        Error_MRI=RMSE_modi(Zhat_MRI,X_image);
        similarity_index=[];
        for i =1:1:q
            mssim=ssim(abs(Zhat_MRI(:,:,i)/max(max(Zhat_MRI(:,:,i)))),abs(X_image(:,:,i)/max(max(X_image(:,:,i)))));
            similarity_index(i)=mssim;
        end
        sim_MRI=min(similarity_index)
        %save('C:\Users\sbabu\Desktop\Result\brain_8\Xhat_MGD_MEC.mat', 'Xhat_GD_MEC');
        fprintf(fid, '%s(%d) & %8.4f (%5.2f)& %8.4f (%5.2f) \n', name, radial(ii),Error_MRI,Time_MRI,Error_MRI2,Time_MRI2);
        fprintf(fid2, '%s(%d) &  %8.4f (%5.2f)& %8.4f (%5.2f)  \n', name, radial(ii),sim_MRI,Time_MRI,sim_MRI2,Time_MRI2);
 
    end
end
fclose(fid);
fclose(fid2);

function y=SoftThresh(x,p)
y=(abs(x)-p).*x./abs(x).*(abs(x)>p);
y(isnan(y))=0;
end
