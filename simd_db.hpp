#include <cstdint>
#include <tuple>
#include <functional>

#define COLUMN(_name, _type) \
    struct col_##_name       \
    {                        \
        using type = _type;  \
        _type data;          \
    }

#define TABLE128(_name, _capacity, ...)                                               \
    struct tab_##_name : public simd_db::static_table<64, 16, _capacity, __VA_ARGS__> \
    {                                                                                 \
    };                                                                                \
    tab_##_name table_##_name
#define TABLE256(_name, _capacity, ...)                                               \
    struct tab_##_name : public simd_db::static_table<64, 32, _capacity, __VA_ARGS__> \
    {                                                                                 \
    };                                                                                \
    tab_##_name table_##_name
#define TABLE512(_name, _capacity, ...)                                               \
    struct tab_##_name : public simd_db::static_table<64, 64, _capacity, __VA_ARGS__> \
    {                                                                                 \
    };                                                                                \
    tab_##_name table_##_name

#define VIEW(_name, ...) auto _name = std::make_tuple(__VA_ARGS__)

namespace simd_db
{
    namespace detail
    {
        std::size_t constexpr round_multiple(std::size_t number, std::size_t multiple)
        {
            return ((number + multiple - 1) / multiple) * multiple;
        }
        std::size_t constexpr static_column_allocation_size(std::size_t vec_size, std::size_t capacity, std::size_t type_size)
        {
            return round_multiple(round_multiple(capacity * type_size, vec_size), type_size) / type_size;
        }

        template <std::size_t CACHE_LINE, std::size_t VEC_SIZE, std::size_t CAPACITY, typename T>
        struct static_column
        {
            using type = T;
            alignas(CACHE_LINE) T data[static_column_allocation_size(VEC_SIZE, CAPACITY, sizeof(T))];
            T &operator[](std::size_t index)
            {
                return data[index];
            }
        };
    };

    template <std::size_t CACHE_LINE, std::size_t VEC_SIZE, std::size_t CAPACITY, typename... Ts>
    class static_table
    {
        template <typename T>
        using column_template = detail::static_column<CACHE_LINE, VEC_SIZE, CAPACITY, T>;

        std::size_t _size = 0;
        std::tuple<column_template<Ts>...> _columns;

        template <std::size_t... Is>
        void destroy_impl(std::size_t i, std::index_sequence<Is...>)
        {
            _size--;
            (..., (std::get<Is>(_columns)[i] = std::get<Is>(_columns)[_size]));
        }

    public:
        template <typename T>
        auto *column()
        {
            return reinterpret_cast<typename T::type *>(std::get<column_template<T>>(_columns).data);
        }

        template <typename T>
        auto &create()
        {
            return std::get<column_template<T>>(_columns)[_size].data;
        }

        std::size_t create()
        {
            return _size++;
        }

        void destroy(std::size_t i)
        {
            destroy_impl(i, std::make_index_sequence<sizeof...(Ts)>{});
        }

        std::size_t size() const
        {
            return _size;
        }

        template <typename T>
        static constexpr std::size_t v_step()
        {
            return VEC_SIZE / sizeof(typename T::type);
        }
    };

};