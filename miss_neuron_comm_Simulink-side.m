function a_y_nn = fcn(Stamper,R,azim)
persistent stamp
if isempty(stamp)
    stamp = 0;
end
Stamper = stamp;
comufile = fopen('link.combuff','w');
fprintf(comufile,'%f,%f,%f,',Stamper,R,azim);
fclose(comufile);
% 'waiting for python link'
com = [0,0,0,0,0,0,0];
while(1)
	comufile = fopen('link.combuff','r');
	in1 = [0,0,0,0,0,0];
    readchar = fread(comufile,'*char');
    readchar = readchar';
%     readchar
    commapos = strfind(readchar,",");
%     commapos
    indexcnt = 1;
    if length(commapos) >= 5
        com(2:7)   = commapos(1:6);
        com(1)     = 0;
        for a = 0:1:5
            in1(indexcnt) = real(str2double(readchar((com(a+1)+1):(com(a+2)-1))));
            indexcnt = indexcnt+1;
        end
        %round(Stamper)
        %in1
        if abs(round(Stamper)-round(in1(4)))<0.0001
            fclose(comufile);
            %'break'
            break
        end
    end
   
%     in1;
end
%'breaked?'
stamp = stamp+1;
a_y_nn  = in1(5);

% this code should be in simulink matlab function block
