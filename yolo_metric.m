clc;clear;
ground_truth = readtable('all_val_gt_50_50.csv');
detected = readtable('Detection_results_50_50.csv');

sorted_gt = sortrows(ground_truth,'image');
sorted_det = sortrows(detected,'image');

det_size = size(detected);
gt_size = size(ground_truth);
prev = 1;
i = 1;
j = 1;
same_count = 0;
% loop though guesses
%
while i <= det_size(1)

    j = prev;
    
    % get the detected image name
    %
    str2 = sorted_det.image{i};
    same_count = 0;
    % loop though the ground truth vals
    %
    while j <= gt_size(1)
        
        if j == 0
            j = 1;
        end
        
        % get ground truth image name
        %
        
        str1 = sorted_gt.image{j};
        
        % check to see if we don't need to keep iterating
        %
        if string(str1) <= string(str2)
        
            % check if the guess is made on the same image as the ground truth
            % labels
            %
            if (strcmp(str1,str2)) 
                if j > 1
                    if str1 == sorted_gt.image{j-1}
                       same_count = same_count + 1; 
                    end
                end
                % ground truth label
                %
                gt_coords = [sorted_gt.xmin(j),sorted_gt.ymin(j),sorted_gt.xmax(j),sorted_gt.ymax(j)];
                
                % guessed label
                %
                det_coords = [sorted_det.xmin(i),sorted_det.ymin(i),sorted_det.xmax(i),sorted_det.ymax(i)];

                % intersection / union
                %
                calc = iou(gt_coords,det_coords);
                
                % read in the iou value ( if there is one )
                %
                curr = sorted_det.iou(i);
                
                % check if there isn't an iou value
                %
                if (isnan(curr))
                    sorted_det.iou(i) = calc;
                end
                
                % check if the iou should be updated
                %
                if(sorted_det.iou(i) < calc)
                    sorted_det.iou(i) = calc;
                end

            end
            
            j = j+1;
        else
            
            % save prev index so we don't have to loop through the whol
            % array every time
            %
            prev = j-same_count-1;
            
            % exit while loop
            %
            j = gt_size(1)+1;
            
        end
    end
    i = i+1;
end

false_positives_tot = 0;
false_positives_nan = 0;
false_positives_zero = 0;

false_negatives = 0;

k = 1;
iou_sum = 0;

p_p_image = sorted_det.image(1);
gt_p_image = sorted_det.image(1);
image_count_p = 1;
image_count_gt = 1;
true_positives = 0;

p = zeros(1,det_size(1));
r = zeros(1,det_size(1));
while k <= det_size(1)
    
    curr_imag_p = sorted_det.image(k);
    
    if (~strcmp(sorted_det.image(k),p_p_image))
        image_count_p = image_count_p + 1;
    end
    
    if k < gt_size
        curr_image_gt = sorted_gt.image(k);
        if (~strcmp(sorted_gt.image(k),gt_p_image))
            image_count_gt = image_count_gt + 1;
        end
    end
    
    curr = sorted_det.iou(k);
    if isnan(curr) 
        false_positives_nan = false_positives_nan + 1;
    elseif curr == 0
        false_positives_zero = false_positives_zero + 1;
    elseif curr < 0.50
        false_positives_zero = false_positives_zero + 1;
        
    else
        true_positives = true_positives + 1;
        iou_sum = iou_sum + curr;
    end
    
    p(k) = true_positives / (true_positives +( false_positives_nan + false_positives_zero));
    
    temp = true_positives / gt_size(1);% (true_positives + false_positives_zero);
    if isnan(temp)
        temp = 0;
    end
    r(k) = temp;
    
    
    p_p_image = curr_imag_p;
    gt_p_image = curr_imag_p;
    k = k + 1;

end

false_positives_tot = false_positives_nan + false_positives_zero;
    
false_negatives = abs(gt_size(1) - (det_size(1) - false_positives_nan));


precision = true_positives / (true_positives + false_positives_tot);
recall = true_positives / (true_positives  + false_negatives);


gt_annot_num = 849;
mean_iou = iou_sum / (det_size(1) -false_positives_tot);

x =  linspace(0,890,11);

thing = 1 - recall ;
f1 = 2 * (precision * recall)/(precision+recall);

%cftool(r, p)


%AP = (p1(1)+p1(2)+p1(3)+p1(4)+p1(5)+p1(6)+p1(7)) ;
plot(r,p);
xlabel('recall')
ylabel('precision')

AP =  trapz(r, p);


%[ap1, recall1, precision1] = evaluateDetectionPrecision(detected, ground_truth);


