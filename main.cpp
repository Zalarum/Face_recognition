#include <SFML/Graphics.hpp>
#include <opencv2/opencv.hpp>

#include <random>
#include <cmath>
#include <vector>
#include <iomanip>

#define number_of_models 2

#define number_of_histogram_columns 16

#define width_after_resize 12
#define height_after_resize 14

#define amount_of_points 350

#define show_speed 0.3

using namespace std;
using namespace cv;
using namespace sf;

void extract_histogram(Mat input_photo, vector<int>& histogram_feature)
{
    for (int i = 0; i < input_photo.rows; ++i)
    {
        for (int j = 0; j < input_photo.cols; ++j)
        {
            histogram_feature.at(int(input_photo.at<uchar>(i, j)) / number_of_histogram_columns)++;
        }
    }
}

void extract_resize(Mat input_photo, vector<int>& resize_feature)
{
    resize(input_photo, input_photo, Size(width_after_resize, height_after_resize), INTER_LINEAR);

    for (int i = 0; i < input_photo.rows; ++i)
    {
        for (int j = 0; j < input_photo.cols; ++j)
        {
            resize_feature.push_back(int(input_photo.at<uchar>(i, j)));
        }
    }
}

void extract_random(Mat input_photo, vector<int>& random_feature)
{
    default_random_engine generator;
    uniform_int_distribution<int> x1(0, 111);
    uniform_int_distribution<int> y1(0, 91);

    for (int i = 0; i < amount_of_points; ++i)
    {
        random_feature.push_back(int(input_photo.at<uchar>(x1(generator), y1(generator))));
    }
}

void download_feature_images(vector<vector<vector<int>>>& base_of_features_of_models)
{
    vector<vector<int>> histogram;
    vector<vector<int>> resize;
    vector<vector<int>> random;

    for (int i = 1; i <= number_of_models; ++i)
    {
        for (int j = 1; j <= 40; ++j)
        {
            string t = "images\\s" + to_string(j) + "\\" + to_string(i) + ".pgm";

            Mat image = imread(t, 0);

            vector<int> histogram_feature(number_of_histogram_columns);
            extract_histogram(image, histogram_feature);
            histogram.push_back(histogram_feature);

            vector<int> resize_feature;
            extract_resize(image, resize_feature);
            resize.push_back(resize_feature);

            vector<int> random_feature;
            extract_random(image, random_feature);
            random.push_back(random_feature);
        }
    }

    base_of_features_of_models.push_back(histogram);
    base_of_features_of_models.push_back(resize);
    base_of_features_of_models.push_back(random);
}

double vector_distance(vector<int>& test, vector<int>& model, int& size)
{
    double sum = 0;

    for (int i = 0; i < size; ++i)
    {
        sum += pow(model.at(i) - test.at(i), 2);
    }

    return sqrt(sum);
}

int find_image(vector<int>& test_feature, vector<vector<int>>& model_feature, int size, vector<int>& vote)
{
    double arg = 999999999, tmp;
    int ind1, ind2, ind3;

    for (int i = 0; i < 40 * number_of_models; ++i)
    {
        tmp = vector_distance(test_feature, model_feature.at(i), size);
        if (tmp < arg)
        {
            arg = tmp;
            ind1 = i;
        }
    }
    arg = 999999999;

    for (int i = 0; i < 40 * number_of_models; ++i)
    {
        tmp = vector_distance(test_feature, model_feature.at(i), size);
        if (tmp < arg && i != ind1)
        {
            arg = tmp;
            ind2 = i;
        }
    }
    arg = 999999999;

    for (int i = 0; i < 40 * number_of_models; ++i)
    {
        tmp = vector_distance(test_feature, model_feature.at(i), size);
        if (tmp < arg && i != ind1 && i != ind2)
        {
            arg = tmp;
            ind3 = i;
        }
    }

    vote.push_back(ind1);
    vote.push_back(ind1);
    vote.push_back(ind1);

    vote.push_back(ind2);
    vote.push_back(ind2);

    vote.push_back(ind3);

    return ind1;
}

int main()
{
    vector<vector<vector<int>>> base_of_features_of_models;
    download_feature_images(base_of_features_of_models);

    RenderWindow window(VideoMode(640, 360), "Face recognition");

    //-------------------------------------------------------------

    Texture test_texture, histogram_texture, resize_texture, random_texture, voting_texture;

    test_texture.loadFromFile("None.png");
    histogram_texture.loadFromFile("None.png");
    resize_texture.loadFromFile("None.png");
    random_texture.loadFromFile("None.png");
    voting_texture.loadFromFile("None.png");

    Sprite test_sprite, histogram_sprite, resize_sprite, random_sprite, voting_sprite;

    test_sprite.setTexture(test_texture);
    test_sprite.setPosition(50, 25);

    histogram_sprite.setTexture(histogram_texture);
    histogram_sprite.setPosition(200, 25);

    resize_sprite.setTexture(resize_texture);
    resize_sprite.setPosition(310, 25);

    random_sprite.setTexture(random_texture);
    random_sprite.setPosition(420, 25);

    voting_sprite.setTexture(voting_texture);
    voting_sprite.setPosition(200, 200);

    //-------------------------------------------------------------

    Font font;
    font.loadFromFile("timesnewromanpsmt.ttf");

    Text text("test image", font, 20);
    text.setFillColor(Color::Black);
    text.setPosition(55, 150);

    Text text1("histogram", font, 20);
    text1.setFillColor(Color::Black);
    text1.setPosition(205, 150);

    Text text2("resize", font, 20);
    text2.setFillColor(Color::Black);
    text2.setPosition(330, 150);

    Text text3("random", font, 20);
    text3.setFillColor(Color::Black);
    text3.setPosition(435, 150);

    Text text4("voting", font, 20);
    text4.setFillColor(Color::Black);
    text4.setPosition(220, 310);

    //-------------------------------------------------------------

    int order_of_img = 1, order_of_dir = 1;
    float acc1 = -40 * number_of_models, acc2 = -40 * number_of_models, acc3 = -40 * number_of_models, acc_all = 0;

    Clock clock;
    float time;

    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed)
                window.close();
        }

        time = clock.getElapsedTime().asSeconds();

        //if (show_speed <= time)
        if (Keyboard::isKeyPressed(Keyboard::Space) && show_speed <= time)
        {
            clock.restart();

            string t = "images\\s" + to_string(order_of_dir) + "\\" + to_string(order_of_img) + ".pgm";
            Mat image = imread(t, 0);
            test_texture.loadFromFile(t);

            vector<int> histogram_feature(number_of_histogram_columns);
            extract_histogram(image, histogram_feature);

            vector<int> resize_feature;
            extract_resize(image, resize_feature);

            vector<int> random_feature;
            extract_random(image, random_feature);

            vector<int> vote;

            int ind = find_image(histogram_feature, base_of_features_of_models.at(0), number_of_histogram_columns, vote);

            t = "images\\s" + to_string(ind % 40 + 1) + "\\" + to_string(ind / 40 + 1) + ".pgm";
            histogram_texture.loadFromFile(t);

            if (ind % 40 + 1 == order_of_dir) acc1++;

            ind = find_image(resize_feature, base_of_features_of_models.at(1), width_after_resize * height_after_resize, vote);

            t = "images\\s" + to_string(ind % 40 + 1) + "\\" + to_string(ind / 40 + 1) + ".pgm";
            resize_texture.loadFromFile(t);

            if (ind % 40 + 1 == order_of_dir) acc2++;

            ind = find_image(random_feature, base_of_features_of_models.at(2), amount_of_points, vote);

            t = "images\\s" + to_string(ind % 40 + 1) + "\\" + to_string(ind / 40 + 1) + ".pgm";
            random_texture.loadFromFile(t);

            if (ind % 40 + 1 == order_of_dir) acc3++;

            int count = 0;
            int index = -1;
            for (int l = 0; l < 18; ++l)
            {
                int k = 1;

                for (int m = l + 1; m < 18; ++m) if (vote.at(l) == vote.at(m)) ++k;

                if (k > count)
                {
                    count = k;
                    index = l;
                }
            }

            ind = vote.at(index);

            t = "images\\s" + to_string(ind % 40 + 1) + "\\" + to_string(ind / 40 + 1) + ".pgm";
            voting_texture.loadFromFile(t);

            if (ind % 40 + 1 == order_of_dir) acc_all++;
            cout << setprecision(4) << acc_all / ((order_of_dir - 1) * 10 + order_of_img) * 100 << "%\n";

            if (order_of_img == 10)
            {
                order_of_dir++;
                order_of_img = 1;
            }
            else order_of_img++;

            if (order_of_dir == 40 && order_of_img == 10) break;
        }

        window.clear(Color::White);

        window.draw(text);
        window.draw(text1);
        window.draw(text2);
        window.draw(text3);
        window.draw(text4);

        window.draw(test_sprite);
        window.draw(histogram_sprite);
        window.draw(resize_sprite);
        window.draw(random_sprite);
        window.draw(voting_sprite);

        window.display();
    }

    cout << acc1 / (400 - 40 * number_of_models) * 100 << "% " << acc2 / (400 - 40 * number_of_models) * 100 << "% " << acc3 / (400 - 40 * number_of_models) * 100 << "% " << acc_all / 4 << "%\n";

    return 0;
}