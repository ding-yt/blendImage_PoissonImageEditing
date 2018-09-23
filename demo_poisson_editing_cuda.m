% demo_poisson_editing.m
%
%
% Driver script for demonstration of Poisson image editing for seamless
% cloning of a source image onto a target.
%
%
% Demo cases
% ==========
%
% case 1: airplane (source) over terrain (target)
% case 2: handwritten whiteboard text (source) onto brick wall (target)
%
%
%
%
% References
% ==========
%
% P. Pérez, M. Gangnet, and A. Blake, "Poisson Image Editing," in ACM
% Transactions on Graphics - Proceedings of ACM SIGGRAPH 2003, vol. 22,
% no. 3, pp. 313-318, 2003.  [DOI: 10.1145/882262.882269]
%
%
% Dependencies
% ============
%
% poissonEditingJacobi
% readInputData
% visInputImages
% visEditedImage
% sizeEditRegion
% suptitle
%
%
% --------------------------------------------------
% CompSci 590.06: Parallel Programming
% Department of Computer Science
% Duke University
% --------------------------------------------------
%



%% CLEAN UP

clear variables
close all


%% PARAMETERS

% paths to data and corresponding string descriptors
dirData  = '../data';
pathData = {[dirData filesep 'airplane_terrain.mat'], ...
            [dirData filesep 'writing_wall.mat']};

% brief description for each editing case (corresponding to datasets)
strCase  = {'airplane (source) over terrain (target)', ...
            'written text (source) on brick wall (target)'};

% source coordinates in target domain (top-left corner [i,j]) for each case
ijSrcTopLft = {[ 60, 100], ...
               [ 20,  20]};

% execution parameters
nMaxIter    = 5000;        % max number of Jacobi solver iterations
tauDiffL2   = 1e-6;        % threshold for insignificant iterate difference
methodList = {'original' 'MG1' 'MG3'};  	   % Method choices: original, MG1, MG3



% display flags
flagDisplay      = false; % plot figures/images? (off for shell-only execution)
flagVisSystem    = false; % display editing system data?
stepEchoProgress = 500;  % iteration steps for progress echo


%% (BEGIN)

fprintf( '\n***** BEGIN (%s) *****\n\n', mfilename )


%% INITIALIZATION

% number of demo cases
nCases = length(pathData);

% initialize arrays of empty image-data structs & computation times
Idata(nCases) = struct( 'tgt', [], 'src', [], 'msk', [], 'edited', [] );
tsoln         = zeros( nCases, 1 );

% initialize graphics objects for input/edited image figures
if flagDisplay
    hFig = gobjects( nCases, 1 );
end


%% DEMO CASES (POISSON IMAGE EDITING)

% iterate over all methods
for methodIdx = 1:length(methodList)
    method = methodList{methodIdx};
    
    fprintf( '\n~~~~~~~~ Begin Method: (%s) ~~~~~~~~~~~\n\n',method);

% ~~~ Add profiling ~~~
profile on

% iterate over all cases
for c = 1 : nCases
    
    % prompt next case
%    if c > 1
%        fprintf( '(press any key for next demo case)\n' )
%        pause
%    end
    
    % demo case echo
    fprintf( 'CASE %d: %s\n\n', c, strCase{c} )
    
    % load data
    fprintf( '...loading data...\n' )
    fprintf( '   - path: ''%s''\n', pathData{c} )
    Idata(c) = readInputData( pathData{c} );
    
    % display input data
    if flagDisplay
        hFig(c) = figure;
        visInputImages( hFig(c), Idata(c), ijSrcTopLft{c} );
        suptitle( sprintf( 'case %d images', c ) )
        drawnow
    end
    
    % ~~~ POISSON IMAGE EDITING ~~~
    
    fprintf( '...calling Poisson image editing function...\n' )
    fprintf( '   - source top-left coordinates (i,j) in target: (%d,%d)\n', ...
             ijSrcTopLft{c} )
    fprintf( '   - maximum number of iterations: %d\n', nMaxIter )
    fprintf( '   - relative solution difference threshold: %.2g\n', tauDiffL2 )
 

	% ~~~~ Method choice ~~~~~

	switch lower(method)
    
		case 'original'
			[Idata(c).edited, tsoln(c)] = ...
        	poissonEditingJacobi( Idata(c).tgt, Idata(c).src, Idata(c).msk, ...
                              ijSrcTopLft{c}, nMaxIter, tauDiffL2, ...
                              'stepecho', stepEchoProgress, ...
                              'displaysystem', flagDisplay && flagVisSystem );

		case 'mg1'
			[Idata(c).edited, tsoln(c)] = ...	         
			gpuPoissonEditingJacobi( Idata(c).tgt, Idata(c).src, Idata(c).msk, ...
                              ijSrcTopLft{c}, nMaxIter, tauDiffL2, ...
                              'stepecho', stepEchoProgress, ...
                              'displaysystem', flagDisplay && flagVisSystem );
    
		case 'mg3'
			[Idata(c).edited, tsoln(c)] = ...            
             cudaPoissonEditingJacobi( Idata(c).tgt, Idata(c).src, Idata(c).msk, ...
                               ijSrcTopLft{c}, nMaxIter, tauDiffL2, ...
                               'stepecho', stepEchoProgress, ...
                               'displaysystem', flagDisplay && flagVisSystem );
        otherwise
        error( 'method:UnknownMethod', ...
               ['Unknown poisson editing method: ''' method ''''] )
        
    end  % switch
  
	% echo computation time
    fprintf( ' * elapsed time: %.2fs\n', tsoln(c) )
    
    % display edited image in input-images figure
    if flagDisplay
        visEditedImage( hFig(c), Idata(c).edited );
        drawnow
    end

	% save output image
         hFig(c) = figure;
	set(hFig(c), 'Visible', 'off'); 
        visInputImages( hFig(c), Idata(c), ijSrcTopLft{c} );
         suptitle( sprintf( 'case %d images', c ) )
	visEditedImage( hFig(c), Idata(c).edited );
	saveas(hFig(c),['case_' num2str(c) '_' method '.png']);    
%h = figure;
%    set(h, 'Visible', 'off'); 
%    subplot( 2, 2, 4 )
%    imshow( Idata(c).edited )
%    title( 'edited image' )		
%    saveas(h,['figure_' method '_.png']);
    
    % blank line echo
    fprintf( '\n' )


fprintf( '\n~~~~~~~~ End Method: (%s) ~~~~~~~~~~~\n\n',method);    
end  % for (c; demo cases)


%% (END)

profile off
profileFolder = ['profile_' method]; 
if ~exist(profileFolder, 'dir')
  mkdir(profileFolder);
end
profsave(profile('info'),profileFolder);
%profsave(profile('info'),'/home/home4/yd44/cs590/pj1/poisson-editing/code/profile_jacobian')   
%profsave(profile('info'),'/home/home4/yd44/cs590/pj1/poisson-editing/code/profile_jacobian_mg1')   

fprintf( '\n\nprofiling files in (%s) ready!\n\n', profileFolder);

end  %for (methodList)

fprintf( '\n***** END (%s) *****\n\n', mfilename )


